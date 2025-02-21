import Fastify, { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import WebSocket from 'ws';
import dotenv from 'dotenv';
import fastifyFormBody from '@fastify/formbody';
import fastifyWs from '@fastify/websocket';

// Load environment variables
dotenv.config();

// Check all required environment variables
const requiredEnvVars = ['OPENAI_API_KEY', 'PORT'] as const;
for (const envVar of requiredEnvVars) {
    if (!process.env[envVar]) {
        console.error(`Missing required environment variable: ${envVar}`);
        process.exit(1);
    }
}

interface OpenAISessionUpdate {
    type: 'session.update';
    session: {
        turn_detection: {
            type: string;
            threshold?: number;
            prefix_padding_ms?: number;
            silence_duration_ms?: number;
            create_response?: boolean;
            interrupt_response?: boolean;
        };
        input_audio_format: string;
        output_audio_format: string;
        voice: string;
        instructions: string;
        modalities: string[];
        temperature: number;
    };
}

interface OpenAIConversationItem {
    type: 'conversation.item.create';
    item: {
        type: string;
        role: string;
        content: Array<{
            type: string;
            text: string;
        }>;
    };
}

interface OpenAITruncateEvent {
    type: 'conversation.item.truncate';
    item_id: string;
    content_index: number;
    audio_end_ms: number;
}

interface TwilioMediaEvent {
    event: string;
    media?: {
        timestamp: number;
        payload: string;
    };
    start?: {
        streamSid: string;
    };
}

const { OPENAI_API_KEY } = process.env;
if (!OPENAI_API_KEY) {
    console.error('Missing OpenAI API key. Please set it in the .env file.');
    process.exit(1);
}

// Initialize Fastify with error logging
const fastify: FastifyInstance = Fastify({
    logger: true // Enable Fastify's built-in logging
});
fastify.register(fastifyFormBody);
fastify.register(fastifyWs);

// Constants
const SYSTEM_MESSAGE = 'You are a helpful and bubbly AI assistant ... rickrolling – subtly.';
const VOICE = 'alloy';
const PORT = process.env.PORT || 5050;

const LOG_EVENT_TYPES = [
    'error',
    'response.content.done',
    'rate_limits.updated',
    'response.done',
    'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started',
    'session.created',
] as const;

const SHOW_TIMING_MATH = false;

// Root Route
fastify.get('/', async (_request: FastifyRequest, reply: FastifyReply) => {
    reply.send({ message: 'Twilio Media Stream Server is running!' });
});

// Route for Twilio to handle incoming calls
fastify.all('/voice', async (request: FastifyRequest, reply: FastifyReply) => {
    console.log('Received voice call webhook:', request.body);
    const twimlResponse = `<?xml version="1.0" encoding="UTF-8"?>
    <Response>
      <Say>Please wait while we connect your call to the AI voice assistant...</Say>
      <Pause length="1"/>
      <Say>OK, you can start talking!</Say>
      <Connect>
        <Stream url="wss://${request.headers.host}/media-stream" />
      </Connect>
    </Response>`;
    reply.type('text/xml').send(twimlResponse);
});

// WebSocket route for /media-stream
fastify.register(async (fastify: FastifyInstance) => {
    // NOTE: the 'connection' param is now typed as the raw 'WebSocket' object directly
    fastify.get('/media-stream', { websocket: true }, (connection: WebSocket, req) => {
        console.log('Client connected');

        let streamSid: string | null = null;
        let latestMediaTimestamp = 0;
        let lastAssistantItem: string | null = null;
        let markQueue: string[] = [];
        let responseStartTimestampTwilio: number | null = null;
        let lastSpeechStartTime = 0;

        const openAiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01', {
            headers: {
                Authorization: `Bearer ${OPENAI_API_KEY}`,
                'OpenAI-Beta': 'realtime=v1',
            },
        });

        const initializeSession = (): void => {
            const sessionUpdate: OpenAISessionUpdate = {
                type: 'session.update',
                session: {
                    turn_detection: {
                        type: 'server_vad',
                        threshold: 0.6,                // Increased from default 0.5 for better noise handling
                        prefix_padding_ms: 300,
                        silence_duration_ms: 800,      // Increased from default 200 for longer pauses
                        create_response: true,
                        interrupt_response: true
                    },
                    input_audio_format: 'g711_ulaw',
                    output_audio_format: 'g711_ulaw',
                    voice: VOICE,
                    instructions: SYSTEM_MESSAGE,
                    modalities: ['text', 'audio'],
                    temperature: 0.8,
                },
            };
            console.log('Sending session update:', JSON.stringify(sessionUpdate));
            openAiWs.send(JSON.stringify(sessionUpdate));
        };

        // Example if you want AI to greet first
        const sendInitialConversationItem = (): void => {
            const initialConversation: OpenAIConversationItem = {
                type: 'conversation.item.create',
                item: {
                    type: 'message',
                    role: 'user',
                    content: [
                        {
                            type: 'input_text',
                            text: 'Greet the user with "Hello there! ... How can I help you?"',
                        },
                    ],
                },
            };
            if (SHOW_TIMING_MATH) {
                console.log('Sending initial conversation item:', JSON.stringify(initialConversation));
            }
            openAiWs.send(JSON.stringify(initialConversation));
            openAiWs.send(JSON.stringify({ type: 'response.create' }));
        };

        const handleSpeechStartedEvent = (): void => {
            // Add debounce to prevent rapid start/stop
            if (Date.now() - lastSpeechStartTime < 500) {
                return;
            }
            lastSpeechStartTime = Date.now();

            if (markQueue.length > 0 && responseStartTimestampTwilio !== null) {
                const elapsedTime = latestMediaTimestamp - responseStartTimestampTwilio;
                if (SHOW_TIMING_MATH) {
                    console.log(`Truncation: ${latestMediaTimestamp} - ${responseStartTimestampTwilio} = ${elapsedTime}ms`);
                }
                // Only truncate if significant time has passed
                if (elapsedTime > 1000) {
                    if (lastAssistantItem) {
                        const truncateEvent: OpenAITruncateEvent = {
                            type: 'conversation.item.truncate',
                            item_id: lastAssistantItem,
                            content_index: 0,
                            audio_end_ms: elapsedTime,
                        };
                        if (SHOW_TIMING_MATH) console.log('Sending truncation event:', JSON.stringify(truncateEvent));
                        openAiWs.send(JSON.stringify(truncateEvent));
                    }
                    // Clear Twilio's queued audio
                    connection.send(
                        JSON.stringify({
                            event: 'clear',
                            streamSid: streamSid,
                        })
                    );
                    markQueue = [];
                    lastAssistantItem = null;
                    responseStartTimestampTwilio = null;
                }
            }
        };

        const sendMark = (ws: WebSocket, sSid: string): void => {
            if (sSid) {
                const markEvent = {
                    event: 'mark',
                    streamSid: sSid,
                    mark: { name: 'responsePart' },
                };
                ws.send(JSON.stringify(markEvent));
                markQueue.push('responsePart');
            }
        };

        // OpenAI WebSocket events
        openAiWs.on('open', () => {
            console.log('Connected to OpenAI Realtime API');
            setTimeout(initializeSession, 100);
            // sendInitialConversationItem(); // optional
        });

        openAiWs.on('message', (rawData: WebSocket.Data) => {
            try {
                const response = JSON.parse(rawData.toString());
                if (LOG_EVENT_TYPES.includes(response.type)) {
                    console.log(`OpenAI event: ${response.type}`, response);
                }

                if (response.type === 'response.audio.delta' && response.delta) {
                    const audioDelta = {
                        event: 'media',
                        streamSid: streamSid,
                        media: { payload: response.delta },
                    };
                    connection.send(JSON.stringify(audioDelta));

                    if (!responseStartTimestampTwilio) {
                        responseStartTimestampTwilio = latestMediaTimestamp;
                        if (SHOW_TIMING_MATH) {
                            console.log(`Set responseStartTimestamp: ${responseStartTimestampTwilio}ms`);
                        }
                    }

                    if (response.item_id) {
                        lastAssistantItem = response.item_id;
                    }
                    sendMark(connection, streamSid!);
                }

                if (response.type === 'input_audio_buffer.speech_started') {
                    handleSpeechStartedEvent();
                }
            } catch (err) {
                console.error('Error handling OpenAI message:', err, 'Raw:', rawData);
            }
        });

        // Twilio -> AI
        connection.on('message', (rawMsg: WebSocket.Data) => {
            try {
                const data: TwilioMediaEvent = JSON.parse(rawMsg.toString());
                switch (data.event) {
                    case 'media':
                        if (data.media) {
                            latestMediaTimestamp = data.media.timestamp;
                            if (SHOW_TIMING_MATH) {
                                console.log(`Received Twilio media timestamp: ${latestMediaTimestamp}ms`);
                            }
                            if (openAiWs.readyState === WebSocket.OPEN) {
                                const audioAppend = {
                                    type: 'input_audio_buffer.append',
                                    audio: data.media.payload,
                                };
                                openAiWs.send(JSON.stringify(audioAppend));
                            }
                        }
                        break;
                    case 'start':
                        if (data.start) {
                            streamSid = data.start.streamSid;
                            console.log('Incoming stream started:', streamSid);
                            responseStartTimestampTwilio = null;
                            latestMediaTimestamp = 0;
                        }
                        break;
                    case 'mark':
                        if (markQueue.length > 0) {
                            markQueue.shift();
                        }
                        break;
                    default:
                        console.log('Received non-media event:', data.event);
                        break;
                }
            } catch (err) {
                console.error('Error parsing Twilio message:', err, 'Message:', rawMsg);
            }
        });

        // When the Twilio connection closes
        connection.on('close', () => {
            if (openAiWs.readyState === WebSocket.OPEN) {
                openAiWs.close();
            }
            console.log('Client disconnected.');
        });

        // OpenAI close/error
        openAiWs.on('close', () => {
            console.log('Disconnected from OpenAI Realtime API');
        });

        // Enhanced error logging for OpenAI WebSocket
        openAiWs.on('error', (err: Error) => {
            console.error('OpenAI WebSocket error:', err);
            console.error('OpenAI connection state:', openAiWs.readyState);
            // Try to send an error message to Twilio
            try {
                connection.send(JSON.stringify({
                    event: 'error',
                    error: 'OpenAI connection error'
                }));
            } catch (e) {
                console.error('Failed to send error to Twilio:', e);
            }
        });
    });
});

// Add error handlers
fastify.setErrorHandler((error, request, reply) => {
    request.log.error(error);
    reply.status(500).send({ error: 'Application error occurred' });
});

// Start server
const start = async (): Promise<void> => {
    try {
        await fastify.listen({ port: Number(PORT) });
        console.log(`Server running on port ${PORT}`);
    } catch (err) {
        console.error(err);
        process.exit(1);
    }
};

start();
