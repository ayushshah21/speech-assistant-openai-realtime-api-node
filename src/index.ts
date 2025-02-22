import Fastify, { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import WebSocket from 'ws';
import dotenv from 'dotenv';
import fastifyFormBody from '@fastify/formbody';
import fastifyWs from '@fastify/websocket';
import fs from 'fs';
import path from 'path';

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

// Knowledge base interfaces
interface KnowledgeArticle {
    id: string;
    title: string;
    category: string;
    content: {
        overview: string;
        solution?: string[];
        steps?: string[];
        important_notes?: string | string[];
        [key: string]: any;
    };
    faq: Array<{
        question: string;
        answer: string;
    }>;
}

interface KnowledgeBase {
    articles: KnowledgeArticle[];
    [key: string]: any;
}

// Load knowledge base
const kbPath = path.join(process.cwd(), 'src', 'knowledge_base.json');
const kbData = fs.readFileSync(kbPath, 'utf-8');
const knowledgeBase: KnowledgeBase = JSON.parse(kbData);

/**
 * Creates a concise context from the knowledge base for the AI
 */
function buildKnowledgeBaseContext(): string {
    const articles = knowledgeBase.articles.map(article => {
        const parts = [
            `TITLE: ${article.title}`,
            `OVERVIEW: ${article.content.overview}`
        ];

        // Add solution steps if present
        if (article.content.solution) {
            parts.push(`SOLUTION: ${article.content.solution.join('; ')}`);
        }

        // Add important notes
        if (article.content.important_notes) {
            const notes = Array.isArray(article.content.important_notes)
                ? article.content.important_notes.join('; ')
                : article.content.important_notes;
            parts.push(`NOTES: ${notes}`);
        }

        // Add first FAQ as example
        if (article.faq?.length > 0) {
            const faq = article.faq[0];
            parts.push(`FAQ: Q: ${faq.question} A: ${faq.answer}`);
        }

        return parts.join('\n');
    });

    return articles.join('\n\n');
}

// Build knowledge base context once at startup
const KNOWLEDGE_BASE_CONTEXT = buildKnowledgeBaseContext();

// System message that enforces staying on topic
const SYSTEM_MESSAGE = `You are a Kayako AI support assistant. You have access to the following knowledge base about Kayako's products and services:

${KNOWLEDGE_BASE_CONTEXT}

IMPORTANT GUIDELINES:
1. ONLY answer questions related to Kayako's products and services
2. If a user asks about anything not related to Kayako, respond with:
   "I'm specifically trained to help with Kayako's products and services. What would you like to know about Kayako?"
3. Keep responses concise and friendly
4. Use the knowledge base information to provide accurate answers
5. If you don't find a specific answer in the knowledge base, say:
   "I don't have specific information about that aspect of Kayako. Would you like me to connect you with a support specialist?"

Remember: Your purpose is to help users with Kayako-related questions only.`;

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
const VOICE = 'alloy';
const PORT = process.env.PORT || 5050;
const MAX_CHUNK_SIZE = 8192; // Maximum size for Twilio audio chunks

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
      <Say>Welcome to Kayako support. How can I assist you today?</Say>
      <Connect>
        <Stream url="wss://${request.headers.host}/media-stream" />
      </Connect>
    </Response>`;
    reply.type('text/xml').send(twimlResponse);
});

// WebSocket route for /media-stream
fastify.register(async (fastify: FastifyInstance) => {
    fastify.get('/media-stream', { websocket: true }, (connection: WebSocket, req) => {
        console.log('Client connected');

        let streamSid: string | null = null;
        let latestMediaTimestamp = 0;
        let lastAssistantItem: string | null = null;
        let markQueue: string[] = [];
        let responseStartTimestampTwilio: number | null = null;
        let lastSpeechStartTime = 0;
        let isResponseFullyDone = false;  // Track if response.done has fired

        const openAiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01', {
            headers: {
                Authorization: `Bearer ${OPENAI_API_KEY}`,
                'OpenAI-Beta': 'realtime=v1',
            },
        });

        // Helper function to send silence buffer
        function sendSilence() {
            if (!streamSid) return;
            // 160 frames of 0xFF is ~500ms g711 silence
            const silence = Buffer.alloc(160, 0xFF).toString('base64');

            // Send silence in chunks to avoid Twilio's 8192 byte limit
            let offset = 0;
            while (offset < silence.length) {
                const slice = silence.slice(offset, offset + MAX_CHUNK_SIZE);
                offset += MAX_CHUNK_SIZE;
                connection.send(JSON.stringify({
                    event: 'media',
                    streamSid: streamSid,
                    media: { payload: slice }
                }));
            }

            // Mark the silence
            connection.send(JSON.stringify({
                event: 'mark',
                streamSid: streamSid,
                mark: { name: 'endSilence' }
            }));

            console.log('Sent final silence buffer');
        }

        const initializeSession = (): void => {
            const sessionUpdate: OpenAISessionUpdate = {
                type: 'session.update',
                session: {
                    turn_detection: {
                        type: 'server_vad',
                        threshold: 0.6,                // Higher threshold for better noise handling
                        prefix_padding_ms: 300,
                        silence_duration_ms: 800,      // Longer pause detection for natural conversation
                        create_response: true,
                        interrupt_response: true       // Allow interrupting for better flow
                    },
                    input_audio_format: 'g711_ulaw',
                    output_audio_format: 'g711_ulaw',
                    voice: VOICE,
                    instructions: SYSTEM_MESSAGE,      // Using our enhanced system message with KB context
                    modalities: ['text', 'audio'],
                    temperature: 0.7,                  // Lower temperature for more focused responses
                },
            };
            console.log('Initializing OpenAI session with Kayako knowledge base context');
            openAiWs.send(JSON.stringify(sessionUpdate));
        };

        // OpenAI WebSocket events
        openAiWs.on('open', () => {
            console.log('Connected to OpenAI Realtime API');
            setTimeout(initializeSession, 100);
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
                    }

                    if (response.item_id) {
                        lastAssistantItem = response.item_id;
                    }
                    sendMark(connection, streamSid!);
                }

                // Track response.audio.done
                if (response.type === 'response.audio.done') {
                    console.log('AI audio done => waiting for complete response.done');
                }

                // Handle complete response
                if (response.type === 'response.done') {
                    console.log('AI response fully done => sending final silence');
                    isResponseFullyDone = true;
                    if (streamSid) {
                        // Send final silence buffer
                        sendSilence();
                        // Wait ~1s for Twilio to play it
                        setTimeout(() => {
                            console.log('Final silence played => response complete');
                        }, 1000);
                    }
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

        // Helper function to send marks to Twilio
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

        // Handle speech started events with debounce
        const handleSpeechStartedEvent = (): void => {
            if (Date.now() - lastSpeechStartTime < 500) return;
            lastSpeechStartTime = Date.now();

            if (lastAssistantItem) {
                // Truncate AI response
                openAiWs.send(JSON.stringify({
                    type: 'conversation.item.truncate',
                    item_id: lastAssistantItem,
                    content_index: 0,
                    audio_end_ms: 0
                }));
                console.log('Barge-in: truncated AI response');
                lastAssistantItem = null;

                // Clear Twilio's queued audio
                if (streamSid) {
                    connection.send(JSON.stringify({
                        event: 'clear',
                        streamSid: streamSid
                    }));
                }
            }
        };
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
