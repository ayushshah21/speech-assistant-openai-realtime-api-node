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
const requiredEnvVars = [
    'OPENAI_API_KEY',
    'PORT',
    'KAYAKO_API_URL',
    'KAYAKO_USERNAME',
    'KAYAKO_PASSWORD'
] as const;
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

// Types for OpenAI Realtime
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

// Initialize Fastify
const fastify: FastifyInstance = Fastify({
    logger: true // Enable built-in logging
});
fastify.register(fastifyFormBody);
fastify.register(fastifyWs);

// Constants
const VOICE = 'alloy';
const PORT = process.env.PORT || 5050;
const MAX_CHUNK_SIZE = 8192; // Maximum for Twilio audio

// Utility variables
const SHOW_TIMING_MATH = false;

// User + conversation state
interface UserDetails {
    email?: string;
    name?: string;
    phone?: string;
    hasProvidedEmail: boolean;
}

interface ConversationState {
    transcript: Array<{
        role: 'user' | 'assistant';
        content: string;
        timestamp: number;
    }>;
    userDetails: UserDetails;
    kbMatchFound: boolean;
    requiresHumanFollowup: boolean;
}

// Kayako ticket interface
interface KayakoTicket {
    field_values: {
        product: string;
    };
    status_id: string;
    attachment_file_ids: string;
    tags: string;
    type_id: number;
    channel: string;
    subject: string;
    contents: string;
    assigned_agent_id: string;
    assigned_team_id: string;
    requester_id: string;
    channel_id: string;
    priority_id: string;
    channel_options: {
        cc: string[];
        html: boolean;
    };
}

// Kayako config
const KAYAKO_CONFIG = {
    baseUrl: process.env.KAYAKO_API_URL || 'https://doug-test.kayako.com/api/v1',
    username: process.env.KAYAKO_USERNAME || 'anna.kim@trilogy.com',
    password: process.env.KAYAKO_PASSWORD || 'Kayakokayako1?',
    defaultAgent: '309',
    defaultTeam: '1'
};

// Root route
fastify.get('/', async (_request: FastifyRequest, reply: FastifyReply) => {
    reply.send({ message: 'Twilio Media Stream Server is running!' });
});

// Twilio call route
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

// WebSocket route for Twilio Media
fastify.register(async (fastify: FastifyInstance) => {
    fastify.get('/media-stream', { websocket: true }, (connection: WebSocket, req: FastifyRequest) => {
        console.log('Client connected');

        // Pull in caller info from Twilio
        const callerInfo = {
            phone: (req.body as any)?.Caller || null,
            city: (req.body as any)?.CallerCity || null,
            state: (req.body as any)?.CallerState || null,
            country: (req.body as any)?.CallerCountry || null
        };

        let streamSid: string | null = null;
        let latestMediaTimestamp = 0;
        let lastAssistantItem: string | null = null;
        let markQueue: string[] = [];
        let responseStartTimestampTwilio: number | null = null;
        let lastSpeechStartTime = 0;
        let isResponseFullyDone = false;

        // Initialize conversation state
        const conversationState: ConversationState = {
            transcript: [],
            userDetails: {
                hasProvidedEmail: false,
                phone: callerInfo.phone || undefined,
                name: undefined,
                email: undefined
            },
            kbMatchFound: false,
            requiresHumanFollowup: false
        };

        // This system message includes instructions about collecting email
        const ENHANCED_SYSTEM_MESSAGE = `You are a Kayako AI support assistant. You have access to the following knowledge base about Kayako's products and services:

${KNOWLEDGE_BASE_CONTEXT}

IMPORTANT GUIDELINES:
1. Your FIRST priority is to collect the user's email address. Before answering any question, say:
   "Before I assist you, could you please provide your email address so I can follow up if needed?"
2. Once you receive an email, confirm it by saying:
   "Thank you, I've noted your email as [email]. Now, how can I help you with Kayako?"
3. ONLY after getting the email, proceed with these guidelines:
   - ONLY answer questions related to Kayako's products and services
   - If a user asks about anything not related to Kayako, respond with:
     "I'm specifically trained to help with Kayako's products and services. What would you like to know about Kayako?"
   - Keep responses concise and friendly
   - Use the knowledge base information to provide accurate answers
   - If you don't find a specific answer in the knowledge base, say:
     "I don't have specific information about that aspect of Kayako. I'll have a support specialist follow up with you at [email]."

Remember: Always get the email first, then help with Kayako-related questions only.`;

        const openAiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01', {
            headers: {
                Authorization: `Bearer ${OPENAI_API_KEY}`,
                'OpenAI-Beta': 'realtime=v1',
            },
        });

        // Extract email from any text
        function extractEmail(text: string): string | null {
            const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/;
            const match = text.match(emailRegex);
            return match ? match[0] : null;
        }

        // Send G711 silence
        function sendSilence() {
            if (!streamSid) return;
            const silence = Buffer.alloc(160, 0xFF).toString('base64');

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

            connection.send(JSON.stringify({
                event: 'mark',
                streamSid: streamSid,
                mark: { name: 'endSilence' }
            }));

            console.log('Sent final silence buffer');
        }

        // Add message to transcript
        function addToTranscript(role: 'user' | 'assistant', content: string) {
            console.log(`Adding ${role} message to transcript:`, content);

            if (!content.trim()) {
                console.log('Skipping empty message');
                return;
            }

            conversationState.transcript.push({
                role,
                content: content.trim(),
                timestamp: Date.now()
            });

            console.log('Current transcript length:', conversationState.transcript.length);
            console.log('Latest transcript entry:', conversationState.transcript[conversationState.transcript.length - 1]);

            // If it's user speech, check for email
            if (role === 'user' && !conversationState.userDetails.hasProvidedEmail) {
                const email = extractEmail(content);
                if (email) {
                    console.log('Found email in user message:', email);
                    conversationState.userDetails.email = email;
                    conversationState.userDetails.hasProvidedEmail = true;
                }
            }
        }

        // Check if AI response indicates no KB match
        function checkForNoKBMatch(response: string): boolean {
            const noMatchPhrases = [
                "I don't have specific information",
                "I'll have a support specialist follow up",
                "would you like me to connect you with a support specialist",
                "I don't have that information in my knowledge base"
            ];
            return noMatchPhrases.some(phrase => response.toLowerCase().includes(phrase.toLowerCase()));
        }

        // Initialize Realtime session
        const initializeSession = (): void => {
            const sessionUpdate: OpenAISessionUpdate = {
                type: 'session.update',
                session: {
                    turn_detection: {
                        type: 'server_vad',
                        threshold: 0.6,
                        prefix_padding_ms: 300,
                        silence_duration_ms: 800,
                        create_response: true,
                        interrupt_response: true
                    },
                    input_audio_format: 'g711_ulaw',
                    output_audio_format: 'g711_ulaw',
                    voice: VOICE,
                    instructions: ENHANCED_SYSTEM_MESSAGE,
                    modalities: ['text', 'audio'],
                    temperature: 0.7,
                },
            };
            console.log('Initializing OpenAI session with enhanced system message');
            openAiWs.send(JSON.stringify(sessionUpdate));
        };

        // Handle OpenAI WS
        openAiWs.on('open', () => {
            console.log('Connected to OpenAI Realtime API');
            setTimeout(initializeSession, 100);
        });

        openAiWs.on('message', (rawData: WebSocket.Data) => {
            try {
                const response = JSON.parse(rawData.toString());

                console.log('OpenAI event:', response.type, response);

                // If user speech recognized
                if (response.type === 'speech.phrase' && response.text) {
                    addToTranscript('user', response.text);
                }

                // If AI text
                if (response.type === 'response.text' && response.text) {
                    addToTranscript('assistant', response.text);

                    if (checkForNoKBMatch(response.text)) {
                        conversationState.kbMatchFound = false;
                        conversationState.requiresHumanFollowup = true;
                        console.log('No KB match found => needs human followup');
                    } else {
                        conversationState.kbMatchFound = true;
                    }
                }

                // [FIXED HERE] If final transcript is from the AI's audio output
                if (response.type === 'response.audio_transcript.done' && response.transcript) {
                    console.log('Got final assistant transcript from audio:', response.transcript);
                    addToTranscript('assistant', response.transcript);
                }

                // Audio deltas => forward to Twilio
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

                if (response.type === 'response.done') {
                    console.log('AI response fully done => sending final silence');
                    isResponseFullyDone = true;
                    if (streamSid) {
                        sendSilence();
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

        // Connection closed
        connection.on('close', async () => {
            console.log('Call ended, conversation state:', JSON.stringify(conversationState, null, 2));

            // If we have a transcript
            if (conversationState.transcript.length > 0) {
                console.log(`Creating Kayako ticket with ${conversationState.transcript.length} messages`);
                try {
                    await createKayakoTicket(conversationState);
                    console.log('Successfully created Kayako ticket');
                } catch (error) {
                    console.error('Failed to create Kayako ticket:', error);
                    // For debugging
                    console.error('Conversation state at time of failure:', JSON.stringify(conversationState, null, 2));
                }
            } else {
                console.warn('Call ended with no transcript. Possibly no speech recognized or no user input?');
                console.warn('Final conversation state:', JSON.stringify(conversationState, null, 2));
            }

            if (openAiWs.readyState === WebSocket.OPEN) {
                openAiWs.close();
            }
            console.log('Client disconnected.');
        });

        // openAiWs close/error
        openAiWs.on('close', () => {
            console.log('Disconnected from OpenAI Realtime API');
        });

        openAiWs.on('error', (err: Error) => {
            console.error('OpenAI WebSocket error:', err);
            console.error('OpenAI connection state:', openAiWs.readyState);
            try {
                connection.send(JSON.stringify({
                    event: 'error',
                    error: 'OpenAI connection error'
                }));
            } catch (e) {
                console.error('Failed to send error to Twilio:', e);
            }
        });

        // Helper to send "mark" events
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

        // Barge-in
        const handleSpeechStartedEvent = (): void => {
            if (Date.now() - lastSpeechStartTime < 500) return;
            lastSpeechStartTime = Date.now();

            if (lastAssistantItem) {
                openAiWs.send(JSON.stringify({
                    type: 'conversation.item.truncate',
                    item_id: lastAssistantItem,
                    content_index: 0,
                    audio_end_ms: 0
                }));
                console.log('Barge-in: truncated AI response');
                lastAssistantItem = null;

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

// Error handlers
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

/**
 * We changed the email extraction so it can pick up from
 * the entire transcript, not just user messages.
 */
function extractEmailFromMessages(messages: string[]): string | null {
    for (const msg of messages) {
        const match = msg.match(/[\w.-]+@[\w.-]+\.\w+/);
        if (match) {
            return match[0];
        }
    }
    return null;
}

/**
 * Generate a summarized subject, summary, and detect an email from ANY transcript line
 * (not just user lines).
 */
function generateTicketSummary(conversation: ConversationState): {
    subject: string;
    summary: string;
    email: string | null;
    requiresFollowup: boolean;
} {
    // Gather ALL messages from the entire transcript
    const allMessages = conversation.transcript.map(e => e.content);
    const userMessages = conversation.transcript
        .filter(e => e.role === 'user')
        .map(e => e.content);
    const aiResponses = conversation.transcript
        .filter(e => e.role === 'assistant')
        .map(e => e.content);

    // Extract email from the entire transcript or user details
    const email = conversation.userDetails.email || extractEmailFromMessages(allMessages);

    // Check if followup is needed
    const requiresFollowup = aiResponses.some(resp =>
        resp.toLowerCase().includes('specialist') ||
        resp.toLowerCase().includes('follow up') ||
        resp.toLowerCase().includes('don\'t have specific information')
    );

    // Determine subject from the first meaningful user message
    let subject = 'New Support Call';
    if (userMessages.length > 0) {
        const firstMeaningful = userMessages.find(msg => !msg.match(/^[\w.-]+@[\w.-]+\.\w+$/));
        if (firstMeaningful) {
            subject = firstMeaningful.substring(0, 100);
        }
    }

    // Generate a concise summary of the conversation
    const summary = `The customer inquired about ${subject.toLowerCase()}. ${aiResponses.length > 0
        ? `The AI provided assistance with ${generateKeyPoints(conversation.transcript)}. `
        : ''
        }${requiresFollowup
            ? 'The conversation requires human follow-up as the AI could not fully address the inquiry.'
            : 'The AI successfully resolved the customer\'s inquiry.'
        }`;

    return {
        subject,
        summary,
        email,
        requiresFollowup
    };
}

function generateKeyPoints(transcript: Array<{ role: string; content: string }>): string {
    const keyPoints = new Set<string>();
    transcript.forEach(entry => {
        const text = entry.content.toLowerCase();

        if (text.includes('password') && text.includes('reset')) {
            keyPoints.add('Password Reset Assistance');
        }
        if (text.includes('account') || text.includes('login')) {
            keyPoints.add('Account Management');
        }
        if (text.includes('error') || text.includes('issue') || text.includes('problem')) {
            keyPoints.add('Technical Support');
        }
        if (text.includes('how to') || text.includes('how do i')) {
            keyPoints.add('Feature Usage Guidance');
        }
    });

    const result = Array.from(keyPoints).join(', ');
    return result || 'General Inquiry';
}

// Create Kayako ticket
async function createKayakoTicket(conversation: ConversationState): Promise<void> {
    console.log('Starting Kayako ticket creation...');

    if (!conversation.transcript || conversation.transcript.length === 0) {
        throw new Error('Cannot create ticket: No transcript');
    }

    try {
        console.log('Generating ticket summary...');
        const ticketInfo = generateTicketSummary(conversation);
        console.log('Generated ticket info:', ticketInfo);

        // If we extracted an email from the entire transcript
        if (ticketInfo.email && !conversation.userDetails.email) {
            conversation.userDetails.email = ticketInfo.email;
            conversation.userDetails.hasProvidedEmail = true;
        }
        if (ticketInfo.requiresFollowup) {
            conversation.requiresHumanFollowup = true;
        }

        // Format the transcript in chronological order
        const transcriptText = conversation.transcript
            .map(entry => {
                const role = entry.role === 'assistant' ? 'Agent' : 'User';
                return `${role}: ${entry.content}`;
            })
            .join('\n\n');

        // Create the ticket content with proper HTML formatting
        const ticketContent = `
<div style="font-family: Arial, sans-serif; line-height: 1.6;">
    <div style="margin-bottom: 20px;">
        <strong>SUBJECT</strong><br>
        ${ticketInfo.subject}
    </div>

    <div style="margin-bottom: 20px;">
        <strong>SUMMARY</strong><br>
        ${ticketInfo.summary}
    </div>

    <div style="margin-bottom: 20px;">
        <strong>PRIORITY ESTIMATE</strong><br>
        ${ticketInfo.requiresFollowup ? 'HIGH' : 'LOW'}
    </div>

    <div style="margin-bottom: 20px;">
        <strong>CALL TRANSCRIPT</strong><br>
        <div style="white-space: pre-wrap; font-family: monospace;">
${transcriptText}
        </div>
    </div>
</div>`;

        const ticket: KayakoTicket = {
            field_values: {
                product: "80"
            },
            status_id: "1",
            attachment_file_ids: "",
            tags: "gauntlet-ai",
            type_id: 7,
            channel: "MAIL",
            subject: ticketInfo.subject,
            contents: ticketContent,
            assigned_agent_id: KAYAKO_CONFIG.defaultAgent,
            assigned_team_id: KAYAKO_CONFIG.defaultTeam,
            requester_id: KAYAKO_CONFIG.defaultAgent,
            channel_id: "1",
            priority_id: ticketInfo.requiresFollowup ? "2" : "1",
            channel_options: {
                cc: [],
                html: true
            }
        };

        console.log('Sending ticket to Kayako API:', JSON.stringify(ticket, null, 2));

        const response = await fetch(`${KAYAKO_CONFIG.baseUrl}/cases`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Basic ' + Buffer.from(`${KAYAKO_CONFIG.username}:${KAYAKO_CONFIG.password}`).toString('base64')
            },
            body: JSON.stringify(ticket)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to create Kayako ticket. Status: ${response.status}. Response: ${errorText}`);
        }

        const responseData = await response.json();
        console.log('Kayako ticket created successfully:', responseData);

    } catch (error) {
        console.error('Error in createKayakoTicket:', error);
        throw error;
    }
}
