import Fastify, { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import WebSocket from 'ws';
import dotenv from 'dotenv';
import fastifyFormBody from '@fastify/formbody';
import fastifyWs from '@fastify/websocket';
import fs from 'fs';
import path from 'path';
import { createCallRecorder, CallRecorder } from './call-recorder.js';
import FormData from 'form-data';
import { OpenAI } from 'openai';
import https from 'https';
import http from 'http';
import { transferCallToAgent, getCallSidFromStreamSid, evaluateForwardingWithLLM } from './call-forwarding.js';
import pkg from 'twilio';
import os from 'os';
const { Twilio } = pkg;

// Load environment variables
dotenv.config();

// Configuration
const ENABLE_CALL_FORWARDING = process.env.ENABLE_CALL_FORWARDING === 'true';

// Check all required environment variables
const requiredEnvVars = [
    'OPENAI_API_KEY',
    'PORT',
    'KAYAKO_API_URL',
    'KAYAKO_USERNAME',
    'KAYAKO_PASSWORD',
    'TWILIO_ACCOUNT_SID',
    'TWILIO_AUTH_TOKEN'
] as const;
for (const envVar of requiredEnvVars) {
    if (!process.env[envVar]) {
        console.error(`Missing required environment variable: ${envVar}`);
        process.exit(1);
    }
}

// Near the top of the file, add these declarations
const FORWARDING_THRESHOLD = parseInt(process.env.FORWARDING_THRESHOLD || '3', 10);

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
            type: string;               // 'server_vad'
            threshold?: number;         // e.g. 0.5 or 0.6
            prefix_padding_ms?: number; // e.g. 300
            silence_duration_ms?: number; // e.g. 200 or 800
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
const SHOW_TIMING_MATH = false;

// User + conversation state
interface UserDetails {
    email?: string;
    name?: string;
    phone?: string;
    hasProvidedEmail: boolean;
}

enum ConfidenceLevel {
    HIGH = 'high',
    MEDIUM = 'medium',
    LOW = 'low'
}

interface SpeechSegment {
    id: string;
    startTime: number;
    endTime: number | null;
    transcription: string;
    confidence: number;
    isFinal: boolean;
    audioPayloads: string[];
}

interface ConversationState {
    transcript: Array<{
        role: 'user' | 'assistant';
        content: string;
        timestamp: number;
        confidence?: number;
        level?: ConfidenceLevel;
    }>;
    rawUserInput: Array<{
        content: string;
        timestamp: number;
        confidence?: number;
        is_final?: boolean;
        duration?: number;
        start_time?: string;
        end_time?: string;
        level?: ConfidenceLevel;
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
    attachment_file_ids: string[];
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

// Add this interface with the other interfaces
interface KayakoAttachmentResponse {
    id: string;
    [key: string]: any;
}

// Kayako config
const KAYAKO_CONFIG = {
    baseUrl: process.env.KAYAKO_API_URL || 'https://doug-test.kayako.com/api/v1',
    username: process.env.KAYAKO_USERNAME || 'anna.kim@trilogy.com',
    password: process.env.KAYAKO_PASSWORD || 'Kayakokayako1?',
    defaultAgent: '309',
    defaultTeam: '1'
};

// Helper function to check if AI response indicates no KB match
function checkForNoKBMatch(response: string): boolean {
    const noMatchPhrases = [
        "I don't have specific information",
        "I'll have a support specialist follow up",
        "would you like me to connect you with a support specialist",
        "I don't have that information in my knowledge base"
    ];
    return noMatchPhrases.some(phrase => response.toLowerCase().includes(phrase.toLowerCase()));
}

// Helper: extract email from text
function extractEmail(text: string): string | null {
    // First try to match standard email format
    const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/;
    const match = text.match(emailRegex);
    if (match) {
        return match[0];
    }

    // If no standard email found, try to match spoken email format with more variations
    // Handle formats like:
    // - "philly at gmail dot com"
    // - "philly at gmail period com"
    // - "philly (at) gmail (dot) com"
    // - "philly AT gmail DOT com"
    const spokenEmailRegex = /\b([a-zA-Z0-9._%+-]+)\s+(?:at|\(at\)|@)\s+([a-zA-Z0-9.-]+)\s+(?:dot|\(dot\)|period|\.)\s+([a-zA-Z]{2,})\b/i;
    const spokenMatch = text.match(spokenEmailRegex);
    if (spokenMatch) {
        // Convert "philly at gmail dot com" to "philly@gmail.com"
        return `${spokenMatch[1]}@${spokenMatch[2]}.${spokenMatch[3]}`;
    }

    return null;
}

// Helper: extract email from an array of string messages
function extractEmailFromStringArray(messages: string[]): string | null {
    for (const msg of messages) {
        const email = extractEmail(msg);
        if (email) {
            return email;
        }
    }
    return null;
}

// Helper: extract email from an array of transcript messages
function extractEmailFromMessages(messages: Array<{ role: 'user' | 'assistant', content: string, timestamp: number, confidence?: number, level?: ConfidenceLevel }>): string | null {
    for (const msg of messages) {
        const email = extractEmail(msg.content);
        if (email) {
            return email;
        }
    }
    return null;
}

// Initialize Twilio client
const twilioClient = new Twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN);

// Initialize variables for WebSocket connection
const WHISPER_TRANSCRIPTION_INTERVAL = 3000; // 3 seconds
const FORWARDING_CHECK_INTERVAL = 15000; // 15 seconds between forwarding checks (reduced from 60 seconds)

/**
 * Check if a message contains a request to speak with a human agent
 * 
 * @param message - The message to check
 * @returns boolean indicating if the message contains a request to speak with a human agent
 */
function containsHumanAgentRequest(message: string): boolean {
    if (!message) return false;

    const lowerMessage = message.toLowerCase();
    const humanRequestPhrases = [
        'speak to agent',
        'talk to human',
        'real person',
        'speak with someone',
        'human agent',
        'transfer me',
        'connect me to',
        'speak to a human',
        'talk to a person',
        'speak with a representative',
        'connect me with someone',
        'need a human',
        'want to talk to a human',
        'agent please',
        'representative',
        'speak to support',
        'talk to support',
        'human support',
        'need help from a person',
        'can i speak to',
        'can i talk to',
        'talk to your humans',
        'talk to humans',
        'speak with humans',
        'connect to a human',
        'human operator',
        'live agent',
        'live person',
        'actual person',
        'talk to someone else',
        'speak to someone else',
        'get me a human',
        'i want a human',
        'human please'
    ];

    return humanRequestPhrases.some(phrase => lowerMessage.includes(phrase));
}

/**
 * Process audio with Whisper API for more accurate transcription
 * 
 * @param audioBuffer - Array of audio buffers to process
 * @returns Transcription text or null if processing failed
 */
async function processAudioWithWhisper(audioBuffer: Buffer[]): Promise<string | null> {
    if (!audioBuffer.length) return null;

    try {
        // Create a temporary file for the audio
        const tempFilePath = path.join(os.tmpdir(), `whisper-${Date.now()}.wav`);

        // G711 ulaw to linear PCM conversion table (defined at the top of the file)
        // Using the existing ULAW_TO_LINEAR table from the module scope

        // Convert audio buffers to PCM
        const pcmBuffers = audioBuffer.map(buffer => {
            // Convert ulaw to PCM
            const pcmData = Buffer.alloc(buffer.length * 2);
            for (let i = 0; i < buffer.length; i++) {
                // Simple ulaw to PCM conversion
                const sample = buffer[i] === 0xFF ? 0 : (buffer[i] ^ 0xFF) << 2;
                pcmData.writeInt16LE(sample, i * 2);
            }
            return pcmData;
        });

        // Combine PCM buffers
        const combinedPcm = Buffer.concat(pcmBuffers);

        // Create WAV header
        const wavHeader = Buffer.alloc(44);

        // RIFF header
        wavHeader.write('RIFF', 0);
        wavHeader.writeUInt32LE(36 + combinedPcm.length, 4);
        wavHeader.write('WAVE', 8);

        // fmt subchunk
        wavHeader.write('fmt ', 12);
        wavHeader.writeUInt32LE(16, 16);
        wavHeader.writeUInt16LE(1, 20); // PCM format
        wavHeader.writeUInt16LE(1, 22); // Mono
        wavHeader.writeUInt32LE(8000, 24); // Sample rate
        wavHeader.writeUInt32LE(8000 * 2, 28); // Byte rate
        wavHeader.writeUInt16LE(2, 32); // Block align
        wavHeader.writeUInt16LE(16, 34); // Bits per sample

        // data subchunk
        wavHeader.write('data', 36);
        wavHeader.writeUInt32LE(combinedPcm.length, 40);

        // Write WAV file
        fs.writeFileSync(tempFilePath, Buffer.concat([wavHeader, combinedPcm]));

        // Call Whisper API using the OpenAI instance
        const openaiInstance = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY
        });
        const transcription = await openaiInstance.audio.transcriptions.create({
            file: fs.createReadStream(tempFilePath),
            model: 'whisper-1',
            language: 'en',
            response_format: 'json',
            temperature: 0.0,  // Lower temperature for more accurate transcriptions
            prompt: 'This is a customer support call. The customer may be asking questions or describing technical issues.'  // Context helps improve accuracy
        });

        // Clean up temp file
        fs.unlinkSync(tempFilePath);

        // Check if the transcription contains a request to speak with a human agent
        // This check will be done in the WebSocket handler where we have access to
        // streamSid and callForwardingChecked variables
        if (transcription.text && containsHumanAgentRequest(transcription.text)) {
            console.log('🔍 Whisper detected human agent request:', transcription.text);
            // The actual forwarding will be handled by the caller of this function
        }

        return transcription.text;
    } catch (error) {
        console.error('Error processing audio with Whisper:', error);
        return null;
    }
}

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

// Add new interface for tracking speech segments
let currentSpeechSegment: SpeechSegment | null = null;
let isSpeaking = false;

// WebSocket route for Twilio Media
fastify.register(async (fastify: FastifyInstance) => {
    fastify.get('/media-stream', { websocket: true }, (connection: WebSocket, req: FastifyRequest) => {
        console.log('Client connected');

        // Add call recorder
        let callRecorder: CallRecorder | null = null;

        // Caller info from Twilio
        const callerInfo = {
            phone: (req.body as any)?.Caller || null,
            city: (req.body as any)?.CallerCity || null,
            state: (req.body as any)?.CallerState || null,
            country: (req.body as any)?.CallerCountry || null
        };

        let streamSid: string | null = null;
        let callSid: string | null = null;
        let latestMediaTimestamp = 0;
        let lastAssistantItem: string | null = null;
        let markQueue: string[] = [];
        let responseStartTimestampTwilio: number | null = null;
        let lastSpeechStartTime = 0;
        let isResponseFullyDone = false;
        let callForwardingChecked = false;
        let lastForwardingCheckTime = 0;
        let whisperAudioBuffer: Buffer[] = [];
        let lastWhisperTranscriptionTime = 0;
        const WHISPER_TRANSCRIPTION_INTERVAL = 3000; // 3 seconds

        // Initialize conversation state
        const conversationState: ConversationState = {
            transcript: [],
            rawUserInput: [],
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
   "Hello! To better assist you today, could you please share your email address with me?"

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
   - If the user asks to speak with a human agent at any point, acknowledge their request and say:
     "I understand you'd like to speak with a human agent. I'll connect you with a support specialist right away."

Remember: Always get the email first, then help with Kayako-related questions only. Users can request a human agent at any time by saying "speak to a human" or "talk to an agent".`;

        const openAiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01', {
            headers: {
                Authorization: `Bearer ${OPENAI_API_KEY}`,
                'OpenAI-Beta': 'realtime=v1',
            },
        });

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

        function getConfidenceLevel(score: number): ConfidenceLevel {
            if (score > 0.9) return ConfidenceLevel.HIGH;
            if (score > 0.6) return ConfidenceLevel.MEDIUM;
            return ConfidenceLevel.LOW;
        }

        // Add message to transcript with enhanced validation
        function addToTranscript(role: 'user' | 'assistant', content: string, confidence?: number) {
            console.log('\n=== ADDING TO TRANSCRIPT ===');
            console.log(`Role: ${role}`);
            console.log('Content:', content);
            console.log('Confidence:', confidence);

            if (!content?.trim()) {
                console.log('❌ Empty message, skipping');
                return;
            }

            const level = confidence ? getConfidenceLevel(confidence) : undefined;
            console.log('Confidence Level:', level);

            // Store raw user input with confidence level
            if (role === 'user') {
                conversationState.rawUserInput.push({
                    content: content.trim(),
                    timestamp: Date.now(),
                    confidence,
                    level,
                    is_final: true
                });
                console.log('📝 Added to raw user input');
            }

            // Add to main transcript - now including all user messages
            const lastMessage = conversationState.transcript[conversationState.transcript.length - 1];
            if (lastMessage && lastMessage.role === role && lastMessage.content === content.trim()) {
                console.log('❌ Duplicate message, skipping');
                return;
            }

            const newEntry = {
                role,
                content: content.trim(),
                timestamp: Date.now(),
                confidence,
                level
            };
            conversationState.transcript.push(newEntry);
            console.log('✅ Added to main transcript');

            // For low confidence, add a note
            if (role === 'user' && level === ConfidenceLevel.LOW) {
                console.log('⚠️ Low confidence speech - added with confidence indicator');
            }

            // Check for email in user messages
            if (role === 'user' && !conversationState.userDetails.hasProvidedEmail) {
                const email = extractEmailFromStringArray(conversationState.transcript.map(e => e.content));
                if (email) {
                    console.log('📧 Found email:', email);
                    conversationState.userDetails.email = email;
                    conversationState.userDetails.hasProvidedEmail = true;
                }
            }

            // Check if we should forward the call to a human agent after each AI response
            if (role === 'assistant' && streamSid) {
                // Always check for forwarding after AI responses
                checkAndHandleCallForwarding();
            }

            console.log('Current transcript length:', conversationState.transcript.length);
            console.log('Current raw input length:', conversationState.rawUserInput.length);
            console.log('=== END ADDING TO TRANSCRIPT ===\n');
        }

        // Function to check if call should be forwarded and handle the forwarding
        async function checkAndHandleCallForwarding() {
            // If call forwarding is disabled, skip this check
            if (!ENABLE_CALL_FORWARDING) return;

            console.log('Checking if call should be forwarded...');

            // Add type declaration for the static property
            interface CallForwardingFunction extends Function {
                transferAttempted?: string;
            }

            // Track if a transfer was attempted in this session using typed property
            const typedFunc = checkAndHandleCallForwarding as CallForwardingFunction;
            if (typedFunc.transferAttempted === callSid && callSid !== null) {
                console.log(`Transfer already attempted for call ${callSid}, skipping duplicate forwarding`);
                return;
            }

            // Check the last 7 user messages for explicit human agent requests
            const recentMessages = conversationState.transcript
                .slice(-7)
                .filter(msg => msg.role === 'user')
                .map(msg => msg.content);

            // Check if any recent message contains a human agent request
            const explicitRequest = recentMessages.some(message => containsHumanAgentRequest(message));

            // Also check if the AI's responses indicate forwarding is needed
            const aiResponses = conversationState.transcript
                .slice(-7)
                .filter(msg => msg.role === 'assistant')
                .map(msg => msg.content);

            const aiIndicatesForwarding = aiResponses.some(response => {
                return response.toLowerCase().includes('human agent') ||
                    response.toLowerCase().includes('support specialist') ||
                    response.toLowerCase().includes('transfer you') ||
                    response.toLowerCase().includes('connect you') ||
                    response.toLowerCase().includes('prefer to speak with a human');
            });

            // Log the detection
            if (explicitRequest) {
                console.log('Explicit request for human agent detected');
            }

            if (aiIndicatesForwarding) {
                console.log('AI response indicates forwarding is appropriate');
            }

            // Start with forwarding criteria
            let shouldForward = explicitRequest || aiIndicatesForwarding;

            // If not an explicit request, use LLM to evaluate
            if (!shouldForward && conversationState.transcript.length >= FORWARDING_THRESHOLD) {
                try {
                    const result = await evaluateForwardingWithLLM(conversationState.transcript);
                    console.log('LLM forwarding evaluation result:', result);
                    shouldForward = result.shouldForward;

                    // Additional logging for easier debugging
                    if (shouldForward) {
                        console.log('LLM recommends forwarding because:', result.reason);
                    }
                } catch (error) {
                    console.error('Error evaluating forwarding with LLM:', error);
                }
            }

            // Don't forward if we found a knowledge base match, unless explicitly requested
            if (shouldForward && conversationState.kbMatchFound && !explicitRequest) {
                console.log('KB match found, skipping forwarding unless explicitly requested');
                shouldForward = false;
            }

            if (shouldForward) {
                console.log('Preparing to forward call to human agent...');

                try {
                    // Only attempt to transfer if we have a valid callSid and streamSid
                    if (!callSid) {
                        console.log('Cannot forward call: Call SID not available');
                        return;
                    }

                    if (!streamSid) {
                        console.log('Cannot forward call: Stream SID not available');
                        return;
                    }

                    // Set the flag BEFORE attempting transfer to prevent loops
                    typedFunc.transferAttempted = callSid;

                    // Attempt to transfer the call with non-null values
                    await transferCallToAgent(openAiWs, callSid, streamSid, conversationState, addToTranscript);

                    // Mark that the call requires human followup
                    conversationState.requiresHumanFollowup = true;

                    console.log('Call forwarded successfully');
                } catch (error) {
                    console.error('Error forwarding call:', error);

                    // Even if the transfer failed, don't retry automatically
                    // This prevents the infinite loop
                    console.log('Will not attempt to transfer this call again automatically');
                }
            } else {
                console.log('Call does not need to be forwarded at this time');
            }
        }

        // Initialize Realtime session
        const initializeSession = (): void => {
            const sessionUpdate: OpenAISessionUpdate = {
                type: 'session.update',
                session: {
                    turn_detection: {
                        type: 'server_vad',
                        threshold: 0.3,  // Lower threshold to capture more speech
                        prefix_padding_ms: 500,  // Increased padding for better speech capture
                        silence_duration_ms: 800,  // Longer silence duration for better segmentation
                        create_response: true,
                        interrupt_response: true
                    },
                    input_audio_format: 'g711_ulaw',
                    output_audio_format: 'g711_ulaw',
                    voice: VOICE,
                    instructions: ENHANCED_SYSTEM_MESSAGE,
                    modalities: ['text', 'audio'],
                    temperature: 0.7
                },
            };
            console.log('Initializing OpenAI session with enhanced VAD settings');
            openAiWs.send(JSON.stringify(sessionUpdate));
        };

        // OpenAI WebSocket
        openAiWs.on('open', () => {
            console.log('Connected to OpenAI Realtime API');
            setTimeout(initializeSession, 100);
        });

        openAiWs.on('message', (rawData: WebSocket.Data) => {
            try {
                const response = JSON.parse(rawData.toString());

                // Speech recognition events
                if (response.type === 'speech.phrase') {
                    console.log('\n=== SPEECH RECOGNITION ===');
                    console.log('Raw response:', JSON.stringify(response, null, 2));

                    if (!response.text) {
                        console.log('❌ No text in speech recognition response');
                        return;
                    }

                    console.log('🗣️ User said:', response.text);
                    console.log('Confidence:', response.confidence || 'N/A');
                    console.log('Is final:', response.is_final || false);

                    // Add to rawUserInput regardless of finality
                    conversationState.rawUserInput.push({
                        content: response.text,
                        timestamp: Date.now(),
                        confidence: response.confidence,
                        is_final: response.is_final || false,
                        duration: response.duration,
                        start_time: response.start_time,
                        end_time: response.end_time,
                        level: response.confidence ?
                            (response.confidence > 0.8 ? ConfidenceLevel.HIGH :
                                response.confidence > 0.5 ? ConfidenceLevel.MEDIUM :
                                    ConfidenceLevel.LOW) : undefined
                    });

                    // Only add final transcripts to avoid duplicates
                    if (response.is_final) {
                        addToTranscript('user', response.text, response.confidence);
                        console.log('✅ Added final user transcript');
                    }

                    // Update current speech segment if active
                    if (currentSpeechSegment) {
                        currentSpeechSegment.transcription = response.text;
                        currentSpeechSegment.confidence = response.confidence || 0;
                        currentSpeechSegment.isFinal = response.is_final || false;
                    } else {
                        console.log('⚠️ No active speech segment');
                    }

                    console.log('=== END SPEECH RECOGNITION ===\n');
                }

                // AI response events
                if (response.type === 'response.text' || response.type === 'response.audio_transcript.done') {
                    console.log('\n=== AI RESPONSE ===');
                    const text = response.text || response.transcript;
                    if (text) {
                        console.log('💬 AI:', text);

                        // Only add non-duplicate responses
                        const lastMessage = conversationState.transcript[conversationState.transcript.length - 1];
                        if (!lastMessage || lastMessage.role !== 'assistant' || lastMessage.content !== text) {
                            addToTranscript('assistant', text, response.confidence);
                        } else {
                            console.log('⚠️ Duplicate AI response, skipping');
                        }
                    }
                    console.log('=== END AI RESPONSE ===\n');
                }

                // Record AI audio
                if (response.type === 'response.audio.delta' && response.delta) {
                    if (callRecorder) {
                        callRecorder.addAIAudio(response.delta);
                    }

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

                // Speech started event
                if (response.type === 'input_audio_buffer.speech_started') {
                    console.log('\n🎤 User started speaking');
                    handleSpeechStartedEvent();
                }

            } catch (err) {
                console.error('❌ Error handling OpenAI message:', err);
                console.error('Raw data:', rawData.toString());
            }
        });

        // Twilio -> AI
        connection.on('message', (rawMsg: WebSocket.Data) => {
            try {
                const data: TwilioMediaEvent = JSON.parse(rawMsg.toString());

                // Initialize recorder on call start
                if (data.event === 'start') {
                    console.log('\n=== CALL STARTED ===');
                    console.log('Stream SID:', data.start?.streamSid);
                    streamSid = data.start?.streamSid || null;
                    if (streamSid) {
                        callRecorder = createCallRecorder(streamSid);
                        whisperAudioBuffer = []; // Initialize Whisper buffer
                        lastWhisperTranscriptionTime = Date.now();
                        callForwardingChecked = false; // Reset call forwarding flag for new calls
                        console.log('Reset callForwardingChecked flag for new call');

                        // Try to get the Call SID early
                        getCallSidFromStreamSid(streamSid)
                            .then(sid => {
                                if (sid) {
                                    callSid = sid;
                                    console.log('Retrieved Call SID:', callSid);
                                }
                            })
                            .catch(err => console.error('Error getting Call SID:', err));
                    }
                    console.log('=== END CALL STARTED ===\n');
                }

                // Record user audio
                if (data.event === 'media' && data.media) {
                    latestMediaTimestamp = data.media.timestamp;

                    // Add to recorder
                    if (callRecorder) {
                        callRecorder.addUserAudio(data.media.payload);
                    }

                    // Add to Whisper buffer
                    const audioChunk = Buffer.from(data.media.payload, 'base64');
                    whisperAudioBuffer.push(audioChunk);

                    // Process with Whisper periodically
                    const now = Date.now();
                    if (now - lastWhisperTranscriptionTime > WHISPER_TRANSCRIPTION_INTERVAL && whisperAudioBuffer.length > 0) {
                        lastWhisperTranscriptionTime = now;

                        // Clone and clear buffer
                        const bufferToProcess = [...whisperAudioBuffer];
                        whisperAudioBuffer = [];

                        // Process with Whisper
                        processAudioWithWhisper(bufferToProcess).then(transcription => {
                            if (transcription) {
                                console.log('Whisper transcription:', transcription);

                                // Calculate confidence based on content
                                let confidence = 0.8; // Default confidence

                                // Check for human agent request keywords
                                const humanAgentKeywords = [
                                    'human', 'agent', 'person', 'representative', 'support', 'specialist',
                                    'talk to', 'speak to', 'connect me', 'transfer me', 'real person'
                                ];

                                // Boost confidence if human agent keywords are detected
                                const lowerTranscription = transcription.toLowerCase();
                                const containsHumanRequest = humanAgentKeywords.some(keyword =>
                                    lowerTranscription.includes(keyword)
                                );

                                if (containsHumanRequest) {
                                    console.log('🔍 Potential human agent request detected in transcription');
                                    confidence = 0.95; // Higher confidence for potential human requests
                                }

                                // Add to transcript
                                addToTranscript('user', transcription, confidence);

                                // If human request is detected, force a forwarding check
                                if (containsHumanRequest) {
                                    console.log('🔄 Forcing forwarding check due to potential human agent request');
                                    // Reset the last check time to force a new check
                                    lastForwardingCheckTime = 0;
                                    // Check for forwarding immediately
                                    checkAndHandleCallForwarding();
                                }
                            }
                        });
                    }

                    // Send to OpenAI
                    if (openAiWs.readyState === WebSocket.OPEN) {
                        openAiWs.send(JSON.stringify({
                            type: 'input_audio_buffer.append',
                            audio: data.media.payload,
                        }));
                    }

                    // Check for speech end (silence detection)
                    if (isSpeaking && currentSpeechSegment && currentSpeechSegment.audioPayloads.length > 50) {
                        const lastPayloads = currentSpeechSegment.audioPayloads.slice(-20);
                        const isAllSilence = lastPayloads.every(payload => {
                            const buffer = Buffer.from(payload, 'base64');
                            return buffer.every(byte => byte === 0xFF);
                        });

                        if (isAllSilence) {
                            handleSpeechEndedEvent();
                        }
                    }
                }

            } catch (err) {
                console.error('❌ Error handling Twilio message:', err);
            }
        });

        // When Twilio connection closes
        connection.on('close', async () => {
            console.log('Call ended, conversation state:', JSON.stringify(conversationState, null, 2));

            let mp3Path: string | undefined;

            // Finish recording if we have one
            if (callRecorder) {
                try {
                    mp3Path = await callRecorder.finishRecording();
                    console.log('Recorded call saved to:', mp3Path);
                } catch (error) {
                    console.error('Failed to save call recording:', error);
                }
            }

            if (conversationState.transcript.length > 0) {
                console.log(`Creating Kayako ticket with ${conversationState.transcript.length} messages`);
                try {
                    await createKayakoTicket(conversationState, mp3Path);
                    console.log('Successfully created Kayako ticket');
                } catch (error) {
                    console.error('Failed to create Kayako ticket:', error);
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

        function sendMark(ws: WebSocket, sSid: string): void {
            if (sSid) {
                const markEvent = {
                    event: 'mark',
                    streamSid: sSid,
                    mark: { name: 'responsePart' },
                };
                ws.send(JSON.stringify(markEvent));
                markQueue.push('responsePart');
            }
        }

        function handleSpeechStartedEvent(): void {
            if (Date.now() - lastSpeechStartTime < 500) return;
            lastSpeechStartTime = Date.now();
            isSpeaking = true;

            // Start new speech segment
            currentSpeechSegment = {
                id: Date.now().toString(),
                startTime: Date.now(),
                endTime: null,
                transcription: '',
                confidence: 0,
                isFinal: false,
                audioPayloads: []
            };

            console.log('🎤 Started new speech segment at:', new Date(lastSpeechStartTime).toISOString());

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
        }

        // Add speech end detection
        function handleSpeechEndedEvent(): void {
            if (!isSpeaking || !currentSpeechSegment) {
                console.log('❌ Cannot end speech segment: no active segment');
                return;
            }

            isSpeaking = false;
            const endTime = Date.now();
            const duration = (endTime - currentSpeechSegment.startTime) / 1000;

            // Update the speech segment
            currentSpeechSegment.endTime = endTime;
            currentSpeechSegment.isFinal = true;

            // Add detailed speech segment info to raw input
            const segmentInfo = {
                content: `[Speech segment - Duration: ${duration.toFixed(1)}s]`,
                timestamp: currentSpeechSegment.startTime,
                confidence: currentSpeechSegment.confidence,
                duration: duration,
                start_time: new Date(currentSpeechSegment.startTime).toISOString(),
                end_time: new Date(endTime).toISOString(),
                is_final: true,
                level: getConfidenceLevel(currentSpeechSegment.confidence)
            };

            // Only add to rawUserInput if we don't have a transcription yet
            if (!currentSpeechSegment) return;
            const segment = currentSpeechSegment as SpeechSegment;
            const hasTranscription = conversationState.rawUserInput.some(
                entry => entry.timestamp >= segment.startTime
                    && entry.timestamp <= endTime
                    && typeof entry.content === 'string'
                    && !entry.content.startsWith('[Speech segment')
            );

            if (!hasTranscription) {
                conversationState.rawUserInput.push(segmentInfo);
                console.log('📝 Added speech segment info to raw input (no transcription available)');
            }

            console.log('🛑 Speech segment ended:', {
                id: currentSpeechSegment.id,
                duration: duration.toFixed(1) + 's',
                confidence: currentSpeechSegment.confidence,
                transcription: currentSpeechSegment.transcription || 'No transcription',
                isFinal: currentSpeechSegment.isFinal
            });

            currentSpeechSegment = null;
        }
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
 * Below are your existing helper functions,
 * unchanged except for referencing them in the final code.
 */

/**
 * Extract questions from user messages in the transcript
 */
function extractUserQuestions(transcript: Array<{ role: string; content: string; timestamp?: number }>): string[] {
    const questions: string[] = [];

    // Process each user message
    transcript.filter(entry => entry.role === 'user').forEach(entry => {
        const content = entry.content;

        // Split content into sentences
        const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0);

        // Check each sentence for question patterns
        sentences.forEach(sentence => {
            const trimmed = sentence.trim();

            // Check for question marks
            if (trimmed.includes('?')) {
                questions.push(trimmed);
                return;
            }

            // Check for question words at the beginning
            const questionWords = ['how', 'what', 'why', 'where', 'when', 'who', 'can', 'could', 'would', 'is', 'are', 'do', 'does'];
            const firstWord = trimmed.split(' ')[0].toLowerCase();

            if (questionWords.includes(firstWord)) {
                questions.push(trimmed);
                return;
            }

            // Check for phrases indicating a question
            const questionPhrases = ['tell me about', 'i need to know', 'i want to know', 'explain', 'help me with'];
            if (questionPhrases.some(phrase => trimmed.toLowerCase().includes(phrase))) {
                questions.push(trimmed);
            }
        });
    });

    return questions;
}

/**
 * Analyze the conversation to determine which questions were resolved
 */
async function analyzeConversationResolution(
    transcript: Array<{ role: string; content: string; timestamp?: number }>,
    questions: string[]
): Promise<{
    resolvedQuestions: string[];
    unresolvedQuestions: string[];
    hasPositiveAcknowledgment: boolean;
    hasNegativeResponse: boolean;
    topicKeywords: string[];
}> {
    // If there are no questions or transcript is too short, return default values
    if (questions.length === 0 || transcript.length < 2) {
        return {
            resolvedQuestions: [],
            unresolvedQuestions: [],
            hasPositiveAcknowledgment: false,
            hasNegativeResponse: false,
            topicKeywords: []
        };
    }

    try {
        // Format the transcript for the LLM
        const formattedTranscript = transcript.map(entry =>
            `${entry.role === 'user' ? 'Customer' : 'Agent'}: ${entry.content}`
        ).join('\n');

        // Format the questions for the LLM
        const formattedQuestions = questions.map((q, i) => `Question ${i + 1}: ${q}`).join('\n');

        // Use OpenAI to analyze the conversation
        const openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY
        });
        const response = await openai.chat.completions.create({
            model: "gpt-4-turbo-preview",
            messages: [
                {
                    role: "system",
                    content: `You are an expert conversation analyst for customer support interactions. 
                    Your task is to analyze a conversation transcript between a customer and an AI support agent.
                    Determine which questions were resolved satisfactorily and which ones remain unresolved.
                    Also identify if the customer expressed positive acknowledgment or negative sentiment.
                    Extract key topic keywords related to Kayako products and services.`
                },
                {
                    role: "user",
                    content: `Please analyze this customer support conversation:
                    
                    TRANSCRIPT:
                    ${formattedTranscript}
                    
                    CUSTOMER QUESTIONS:
                    ${formattedQuestions}
                    
                    Provide your analysis in JSON format with these fields:
                    - resolvedQuestions: array of questions that were satisfactorily answered
                    - unresolvedQuestions: array of questions that were not fully addressed
                    - hasPositiveAcknowledgment: boolean indicating if customer expressed satisfaction
                    - hasNegativeResponse: boolean indicating if customer expressed dissatisfaction
                    - topicKeywords: array of keywords related to Kayako products/services mentioned`
                }
            ],
            response_format: { type: "json_object" },
            temperature: 0.1
        });

        // Parse the response
        const content = response.choices[0].message.content || '{}';
        const result = JSON.parse(content);

        console.log('LLM conversation analysis result:', result);

        return {
            resolvedQuestions: result.resolvedQuestions || [],
            unresolvedQuestions: result.unresolvedQuestions || [],
            hasPositiveAcknowledgment: result.hasPositiveAcknowledgment || false,
            hasNegativeResponse: result.hasNegativeResponse || false,
            topicKeywords: result.topicKeywords || []
        };
    } catch (error) {
        console.error('Error analyzing conversation with LLM:', error);

        // Fallback to basic keyword matching if LLM analysis fails
        console.log('Falling back to basic keyword matching for conversation analysis');
        const resolvedQuestions: string[] = [];
        const unresolvedQuestions: string[] = questions.slice();
        let hasPositiveAcknowledgment = false;
        let hasNegativeResponse = false;
        const topicKeywords: string[] = [];

        // Extract keywords from the conversation
        transcript.forEach(entry => {
            const text = entry.content.toLowerCase();

            // Check for keywords related to Kayako topics
            const kayakoKeywords = [
                'sso', 'single sign-on', 'login', 'password', 'reset', 'account',
                'admin', 'administrator', 'user', 'profile', 'email', 'update',
                'ticket', 'support', 'help', 'issue', 'problem', 'error'
            ];

            kayakoKeywords.forEach(keyword => {
                if (text.includes(keyword) && !topicKeywords.includes(keyword)) {
                    topicKeywords.push(keyword);
                }
            });
        });

        // Check for positive acknowledgment in the last few user messages
        const lastUserMessages = transcript
            .filter(entry => entry.role === 'user')
            .slice(-3)
            .map(entry => entry.content.toLowerCase());

        hasPositiveAcknowledgment = lastUserMessages.some(msg =>
            msg.includes('thank') ||
            msg.includes('great') ||
            msg.includes('perfect') ||
            msg.includes('helpful') ||
            msg.includes('appreciate') ||
            msg.includes('got it') ||
            msg.includes('understand') ||
            msg.includes('clear')
        );

        hasNegativeResponse = lastUserMessages.some(msg =>
            msg.includes('not working') ||
            msg.includes('doesn\'t work') ||
            msg.includes('didn\'t work') ||
            msg.includes('doesn\'t help') ||
            msg.includes('didn\'t help') ||
            msg.includes('still have') ||
            msg.includes('still not') ||
            msg.includes('not what i') ||
            msg.includes('not correct')
        );

        return {
            resolvedQuestions,
            unresolvedQuestions,
            hasPositiveAcknowledgment,
            hasNegativeResponse,
            topicKeywords
        };
    }
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

/**
 * Transcribe an audio file using OpenAI's Whisper API
 */
async function transcribeAudioFile(audioPath: string): Promise<string> {
    console.log('Transcribing audio file:', audioPath);

    try {
        const openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY
        });
        const transcription = await openai.audio.transcriptions.create({
            file: fs.createReadStream(audioPath),
            model: "whisper-1",
            language: "en",
            response_format: "text"
        });

        console.log('Successfully transcribed audio file');
        return transcription;
    } catch (error) {
        console.error('Error transcribing audio:', error);
        throw error;
    }
}

/**
 * Parse the audio transcript into structured conversation using OpenAI
 */
async function parseTranscriptWithAI(transcript: string): Promise<Array<{ role: 'agent' | 'customer', text: string }>> {
    const openai = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY
    });

    const response = await openai.chat.completions.create({
        model: "gpt-4-turbo-preview",
        messages: [{
            role: "system",
            content: `You are a conversation parser for customer support calls. Your task is to break down a transcript into 
                     a sequence of alternating messages between agent and customer. 

                     Key patterns to recognize:
                     - Agent typically starts with phrases like "Before I assist you", "Thank you", "I've noted", "How can I help"
                     - Customer responses are usually informal and include questions or provide information
                     - The conversation follows a typical pattern: agent asks for info -> customer provides -> agent confirms
                     
                     IMPORTANT INSTRUCTIONS:
                     1. If the customer provides an email address in any format (standard or spoken like "name at domain dot com"), 
                        preserve it EXACTLY as spoken in the text field.
                     2. If the agent confirms an email address (e.g., "I've noted your email as..."), preserve the EXACT confirmation
                        including the email address as spoken.
                     3. Do not convert spoken email formats (like "name at domain dot com") to standard format ("name@domain.com").
                        Keep them exactly as they appear in the transcript.
                     
                     Format your response as a JSON object with a "messages" array where each element has:
                     - role: either "agent" or "customer"
                     - text: the exact text spoken, with special attention to preserving email addresses exactly as spoken`
        }, {
            role: "user",
            content: `Parse this customer support transcript into a structured conversation, carefully separating agent and customer messages, and preserving any email addresses exactly as spoken: ${transcript}`
        }],
        response_format: { type: "json_object" },
        temperature: 0.1
    });

    try {
        const content = response.choices[0].message.content || '{"messages": []}';
        const result = JSON.parse(content);
        if (!result.messages || !Array.isArray(result.messages) || result.messages.length === 0) {
            console.warn('Failed to parse conversation structure, using fallback parsing');
            // Attempt basic fallback parsing with proper typing
            const messages: Array<{ role: 'agent' | 'customer', text: string }> = transcript
                .split(/(?=[A-Z][a-z]+:)/)
                .filter(Boolean)
                .map(segment => {
                    const role: 'agent' | 'customer' = (
                        segment.toLowerCase().includes('before i assist') ||
                        segment.toLowerCase().includes('thank you') ||
                        segment.toLowerCase().includes('how can i help')
                    ) ? 'agent' : 'customer';
                    return { role, text: segment.trim() };
                });
            return messages;
        }
        return result.messages;
    } catch (error) {
        console.error('Failed to parse AI response:', error);
        return [{
            role: 'customer' as const,
            text: transcript
        }];
    }
}

// Update createKayakoTicket to use async/await with the new async generateTicketSummary
async function createKayakoTicket(conversation: ConversationState, mp3Path?: string): Promise<void> {
    console.log('Starting Kayako ticket creation...');

    if (!conversation.transcript && !conversation.rawUserInput) {
        throw new Error('Cannot create ticket: No transcript or raw input');
    }

    try {
        // Get audio transcript and parse it with AI first, so we can use it for ticket summary
        let audioTranscript: string | undefined;
        let parsedTranscript: Array<{ role: 'agent' | 'customer', text: string }> = [];

        if (mp3Path && fs.existsSync(mp3Path)) {
            try {
                audioTranscript = await transcribeAudioFile(mp3Path);
                console.log('Generated audio transcript:', audioTranscript);

                // Parse transcript with AI
                parsedTranscript = await parseTranscriptWithAI(audioTranscript);
                console.log('Parsed transcript:', parsedTranscript);

                // Special check for spoken email format in the transcript
                const fullTranscriptText = audioTranscript || '';
                const spokenEmailRegex = /\b([a-zA-Z0-9._%+-]+)\s+at\s+([a-zA-Z0-9.-]+)\s+dot\s+([a-zA-Z]{2,})\b/i;
                const spokenMatch = fullTranscriptText.match(spokenEmailRegex);
                if (spokenMatch) {
                    const email = `${spokenMatch[1]}@${spokenMatch[2]}.${spokenMatch[3]}`;
                    console.log('Found spoken email in full transcript:', email);
                    conversation.userDetails.email = email;
                    conversation.userDetails.hasProvidedEmail = true;
                }

                // Update conversation state with the parsed transcript if it's empty or incomplete
                if (conversation.transcript.length <= 4 && parsedTranscript.length > 0) {
                    console.log('Updating conversation state with parsed transcript data');

                    // Clear existing transcript
                    conversation.transcript = [];

                    // Add parsed messages to transcript
                    parsedTranscript.forEach(message => {
                        conversation.transcript.push({
                            role: message.role === 'customer' ? 'user' : 'assistant',
                            content: message.text,
                            timestamp: Date.now() - Math.floor(Math.random() * 60000) // Approximate timestamps
                        });
                    });

                    console.log('Updated conversation transcript with', conversation.transcript.length, 'messages');
                }

                // Extract email from parsed transcript
                if (parsedTranscript.length > 0 && !conversation.userDetails.email) {
                    console.log('Checking parsed transcript for email...');

                    // Helper function to extract email - defined inline to avoid scope issues
                    const extractEmailFromText = (text: string): string | null => {
                        // First try to match standard email format
                        const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/;
                        const match = text.match(emailRegex);
                        if (match) {
                            return match[0];
                        }

                        // If no standard email found, try to match spoken email format with more variations
                        // Handle formats like:
                        // - "philly at gmail dot com"
                        // - "philly at gmail period com"
                        // - "philly (at) gmail (dot) com"
                        // - "philly AT gmail DOT com"
                        const spokenEmailRegex = /\b([a-zA-Z0-9._%+-]+)\s+(?:at|\(at\)|@)\s+([a-zA-Z0-9.-]+)\s+(?:dot|\(dot\)|period|\.)\s+([a-zA-Z]{2,})\b/i;
                        const spokenMatch = text.match(spokenEmailRegex);
                        if (spokenMatch) {
                            // Convert "philly at gmail dot com" to "philly@gmail.com"
                            return `${spokenMatch[1]}@${spokenMatch[2]}.${spokenMatch[3]}`;
                        }

                        return null;
                    };

                    // Check each customer message for an email
                    for (const message of parsedTranscript) {
                        if (message.role === 'customer') {
                            const email = extractEmailFromText(message.text);
                            if (email) {
                                console.log('Found email in parsed transcript:', email);
                                conversation.userDetails.email = email;
                                conversation.userDetails.hasProvidedEmail = true;
                                break;
                            }
                        }
                    }

                    // Also check agent messages that might confirm the email
                    if (!conversation.userDetails.email) {
                        for (const message of parsedTranscript) {
                            if (message.role === 'agent' &&
                                (message.text.includes("I've noted your email as") ||
                                    message.text.includes("noted your email as"))) {
                                const emailMatch = message.text.match(/noted your email as\s+([^.]+)/i);
                                if (emailMatch && emailMatch[1]) {
                                    const potentialEmail = extractEmailFromText(emailMatch[1]);
                                    if (potentialEmail) {
                                        console.log('Found email in agent confirmation:', potentialEmail);
                                        conversation.userDetails.email = potentialEmail;
                                        conversation.userDetails.hasProvidedEmail = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            } catch (error) {
                console.error('Failed to transcribe or parse audio file:', error);
            }
        }

        console.log('Generating ticket summary...');
        const ticketInfo = await generateTicketSummary(conversation, parsedTranscript);
        console.log('Generated ticket info:', ticketInfo);

        // Skip MP3 file upload entirely
        let attachmentId = '';
        console.log('Skipping call recording upload to avoid attachment issues');

        if (ticketInfo.email && !conversation.userDetails.email) {
            conversation.userDetails.email = ticketInfo.email;
            conversation.userDetails.hasProvidedEmail = true;
        }
        if (ticketInfo.requiresFollowup) {
            conversation.requiresHumanFollowup = true;
        }

        // Extract variables needed for ticket content
        const aiResponses = conversation.transcript
            .filter(e => e.role === 'assistant')
            .map(e => e.content);
        const noKBMatch = aiResponses.some((resp: string) => checkForNoKBMatch(resp));
        const hasNegativeResponse = ticketInfo.unresolvedQuestions.length > 0;
        const topicKeywords: string[] = [];

        // Combine transcript and raw input chronologically
        const allInteractions = [
            ...conversation.transcript.map(entry => ({
                type: 'transcript',
                role: entry.role,
                content: entry.content,
                timestamp: entry.timestamp
            })),
            ...conversation.rawUserInput.map(entry => ({
                type: 'raw',
                role: 'user',
                content: entry.content,
                timestamp: entry.timestamp,
                confidence: entry.confidence
            }))
        ].sort((a, b) => a.timestamp - b.timestamp);

        // Format transcript with proper spacing and role labels
        const transcriptText = allInteractions
            .map(entry => {
                const timestamp = new Date(entry.timestamp).toISOString();
                if (entry.type === 'transcript') {
                    const role = entry.role === 'assistant' ? 'Agent' : 'Customer';
                    return `${role} [${timestamp}]: ${entry.content}`;
                } else {
                    // Raw input entries
                    return `Customer (Raw) [${timestamp}] (Confidence: ${(entry as any).confidence?.toFixed(2) || 'N/A'}): ${entry.content}`;
                }
            })
            .join('\n\n');

        // Add resolved and unresolved questions to the ticket content
        const resolvedQuestionsHtml = ticketInfo.resolvedQuestions.length > 0
            ? `<div style="margin-bottom: 20px; background-color: #e6ffed; padding: 15px; border-radius: 4px; border-left: 4px solid #28a745;">
                <strong>✅ RESOLVED QUESTIONS</strong><br>
                <ul style="margin-top: 10px; margin-bottom: 0;">
                    ${ticketInfo.resolvedQuestions.map(q => `<li>${q}</li>`).join('\n')}
                </ul>
              </div>`
            : '';

        const unresolvedQuestionsHtml = ticketInfo.unresolvedQuestions.length > 0
            ? `<div style="margin-bottom: 20px; background-color: #ffebe9; padding: 15px; border-radius: 4px; border-left: 4px solid #d73a49;">
                <strong>❓ UNRESOLVED QUESTIONS</strong><br>
                <ul style="margin-top: 10px; margin-bottom: 0;">
                    ${ticketInfo.unresolvedQuestions.map(q => `<li>${q}</li>`).join('\n')}
                </ul>
              </div>`
            : '';

        // Add customer sentiment section
        const sentimentHtml = `<div style="margin-bottom: 20px; background-color: ${ticketInfo.requiresFollowup
            ? '#fff8c5; border-left: 4px solid #f9c513'
            : '#e6ffed; border-left: 4px solid #28a745'
            }; padding: 15px; border-radius: 4px;">
            <strong>${ticketInfo.requiresFollowup
                ? '⚠️ FOLLOW-UP REQUIRED'
                : '👍 NO FOLLOW-UP NEEDED'
            }</strong><br>
            <p style="margin-top: 10px; margin-bottom: 0;">
                ${ticketInfo.requiresFollowup
                ? `Reason: ${ticketInfo.unresolvedQuestions.length > 0
                    ? 'Unresolved questions remain'
                    : noKBMatch
                        ? 'No knowledge base match found'
                        : hasNegativeResponse
                            ? 'Customer expressed dissatisfaction'
                            : 'General follow-up required'
                }`
                : 'All questions were resolved successfully and the customer expressed satisfaction.'
            }
            </p>
        </div>`;

        // Add topic keywords section if available
        const topicKeywordsHtml = topicKeywords.length > 0
            ? `<div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
                <strong>🔑 KEY TOPICS</strong><br>
                <p style="margin-top: 10px; margin-bottom: 0;">
                    ${topicKeywords.map(keyword => `<span style="display: inline-block; background-color: #e1e4e8; padding: 2px 8px; margin: 2px; border-radius: 12px;">${keyword}</span>`).join(' ')}
                </p>
              </div>`
            : '';

        const ticketContent = `
<div style="font-family: Arial, sans-serif; line-height: 1.6;">
    <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
        <strong>📋 SUBJECT</strong><br>
        <p style="margin-top: 10px; margin-bottom: 0; font-size: 16px;">${ticketInfo.subject}</p>
    </div>

    <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
        <strong>📧 CUSTOMER EMAIL</strong><br>
        <p style="margin-top: 10px; margin-bottom: 0; font-size: 16px;">${ticketInfo.email || 'Not provided'}</p>
    </div>

    <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
        <strong>📝 SUMMARY</strong><br>
        <p style="margin-top: 10px; margin-bottom: 0;">${ticketInfo.summary}</p>
    </div>

    <div style="margin-bottom: 20px; background-color: ${ticketInfo.priority === 'HIGH'
                ? '#ffebe9; border-left: 4px solid #d73a49'
                : ticketInfo.priority === 'MEDIUM'
                    ? '#fff8c5; border-left: 4px solid #f9c513'
                    : '#e6ffed; border-left: 4px solid #28a745'
            }; padding: 15px; border-radius: 4px;">
        <strong>🔔 PRIORITY: ${ticketInfo.priority}</strong><br>
        <p style="margin-top: 10px; margin-bottom: 0;">
            ${ticketInfo.priority === 'HIGH'
                ? 'Requires immediate attention'
                : ticketInfo.priority === 'MEDIUM'
                    ? 'Should be addressed soon'
                    : 'Can be addressed when convenient'
            }
        </p>
    </div>
    
    ${sentimentHtml}
    ${topicKeywordsHtml}
    ${resolvedQuestionsHtml}
    ${unresolvedQuestionsHtml}

    ${parsedTranscript.length > 0 ? `
    <div style="margin-bottom: 20px;">
        <strong>📞 CALL TRANSCRIPT</strong><br>
        <div style="background: #f5f5f5; padding: 15px; border-radius: 4px; margin: 10px 0; max-height: 400px; overflow-y: auto;">
            ${parsedTranscript.map(message => `
                <div style="margin-bottom: 10px; padding: 8px; background-color: ${message.role === 'agent' ? '#e3f2fd' : '#f3e5f5'}; border-radius: 4px;">
                    <span style="color: ${message.role === 'agent' ? '#2c5282' : '#805ad5'}; font-weight: bold;">
                        ${message.role === 'agent' ? 'Agent' : 'Customer'}: 
                    </span>
                    <span>
                        ${message.text}
                    </span>
                </div>
            `).join('\n')}
        </div>
    </div>
    ` : ''}
</div>`;

        // Map priority to Kayako priority IDs
        const priorityIdMap = {
            'HIGH': '3',   // High priority
            'MEDIUM': '2', // Medium priority
            'LOW': '1'     // Low priority
        };

        const ticket: KayakoTicket = {
            field_values: {
                product: "80"
            },
            status_id: "1",
            attachment_file_ids: [], // Empty array instead of using attachmentId
            tags: "gauntlet-ai",
            type_id: 7,
            channel: "MAIL",
            subject: ticketInfo.subject,
            contents: ticketContent,
            assigned_agent_id: KAYAKO_CONFIG.defaultAgent,
            assigned_team_id: KAYAKO_CONFIG.defaultTeam,
            // TODO: Future enhancement - Create a user in Kayako with the provided email
            // Currently using default agent as requester, but ideally we would:
            // 1. Check if a user with this email exists in Kayako
            // 2. If not, create a new user with the email
            // 3. Use that user's ID as the requester_id
            requester_id: KAYAKO_CONFIG.defaultAgent, // Currently using default agent
            channel_id: "1",
            priority_id: priorityIdMap[ticketInfo.priority] || '2', // Default to medium if mapping fails
            channel_options: {
                cc: [],
                html: true
            }
        };

        // If we have the user's email, add it to the cc field so they receive a copy of the ticket
        if (ticketInfo.email) {
            // Don't add to cc field as it's causing API validation errors
            // Instead, we're already displaying it prominently in the ticket content
            console.log(`Email ${ticketInfo.email} will be displayed in ticket content but not added to cc field`);
            // Keep cc as an empty array to avoid validation errors
            ticket.channel_options.cc = [];
        }

        console.log('Sending ticket to Kayako API:', JSON.stringify(ticket, null, 2));

        const response = await fetch(`${KAYAKO_CONFIG.baseUrl}/cases`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json; charset=UTF-8',
                Authorization: 'Basic ' + Buffer.from(`${KAYAKO_CONFIG.username}:${KAYAKO_CONFIG.password}`).toString('base64')
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

/**
 * Generate a descriptive subject line using LLM based on conversation content
 * 
 * @param transcript - The conversation transcript
 * @param parsedTranscript - Optional parsed transcript from audio
 * @returns Promise with a concise, descriptive subject line
 */
async function generateDescriptiveSubjectWithLLM(
    transcript: Array<{ role: string; content: string; timestamp?: number }>,
    parsedTranscript?: Array<{ role: 'agent' | 'customer', text: string }>
): Promise<string> {
    try {
        // Default subject if LLM call fails
        const defaultSubject = 'Kayako Support Conversation';

        // Format the transcript for the LLM
        let formattedTranscript: string;

        // If we have a parsed transcript from audio, use that as it's likely more accurate
        if (parsedTranscript && parsedTranscript.length > 0) {
            formattedTranscript = parsedTranscript
                .map(entry => `${entry.role === 'customer' ? 'Customer' : 'Agent'}: ${entry.text}`)
                .join('\n');
        } else {
            // Otherwise use the regular transcript
            formattedTranscript = transcript
                .map(entry => `${entry.role === 'user' ? 'Customer' : 'Agent'}: ${entry.content}`)
                .join('\n');
        }

        // Use OpenAI to generate a subject
        const openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY
        });

        const response = await openai.chat.completions.create({
            model: "gpt-4-turbo-preview",
            messages: [
                {
                    role: "system",
                    content: `You are an expert at creating concise, descriptive email subject lines for customer support tickets.
                    Your task is to analyze a conversation transcript between a customer and a Kayako support agent.
                    Create a 5-6 word subject line that clearly describes the main topic or question discussed.
                    The subject should be specific enough that someone can understand what the conversation was about.
                    Focus on Kayako-specific topics like SSO, password reset, branding, administration, etc.
                    DO NOT use generic subjects like "Customer Support" or "Help Request".
                    DO NOT include phrases like "Re:" or "Subject:".
                    Just return the subject line text with no additional explanation or formatting.`
                },
                {
                    role: "user",
                    content: `Please create a concise, descriptive subject line (5-6 words) for this customer support conversation:
                    
                    TRANSCRIPT:
                    ${formattedTranscript}`
                }
            ],
            temperature: 0.3,
            max_tokens: 50
        });

        // Get the subject from the response
        const subject = response.choices[0].message.content?.trim() || defaultSubject;

        // Log the generated subject
        console.log('LLM generated subject:', subject);

        // Ensure the subject isn't too long
        if (subject.length > 70) {
            return subject.substring(0, 67) + '...';
        }

        return subject;
    } catch (error) {
        console.error('Error generating subject with LLM:', error);
        return 'Kayako Support Conversation';
    }
}

// Update generateTicketSummary to accept parsedTranscript as an optional parameter
async function generateTicketSummary(
    conversation: ConversationState,
    parsedTranscript?: Array<{ role: 'agent' | 'customer', text: string }>
): Promise<{
    subject: string;
    summary: string;
    email: string | null;
    requiresFollowup: boolean;
    priority: 'HIGH' | 'MEDIUM' | 'LOW';
    resolvedQuestions: string[];
    unresolvedQuestions: string[];
}> {
    const allMessages = conversation.transcript.map(e => e.content);
    const userMessages = conversation.transcript
        .filter(e => e.role === 'user')
        .map(e => e.content);
    const aiResponses = conversation.transcript
        .filter(e => e.role === 'assistant')
        .map(e => e.content);

    // Enhanced email extraction
    let email: string | null = conversation.userDetails.email || null;

    // If no email in conversation state, try to extract from parsed transcript
    if (!email && parsedTranscript && parsedTranscript.length > 0) {
        console.log('Checking parsed transcript for email in generateTicketSummary');

        // Check each customer message for an email
        for (const message of parsedTranscript) {
            if (message.role === 'customer') {
                // Try standard email format
                const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/;
                const match = message.text.match(emailRegex);
                if (match) {
                    email = match[0];
                    console.log('Found standard email in parsed transcript:', email);
                    break;
                }

                // Try spoken email format
                const spokenEmailRegex = /\b([a-zA-Z0-9._%+-]+)\s+at\s+([a-zA-Z0-9.-]+)\s+dot\s+([a-zA-Z]{2,})\b/i;
                const spokenMatch = message.text.match(spokenEmailRegex);
                if (spokenMatch) {
                    email = `${spokenMatch[1]}@${spokenMatch[2]}.${spokenMatch[3]}`;
                    console.log('Found spoken email in parsed transcript:', email);
                    break;
                }
            }
        }

        // If still no email, check agent confirmations
        if (!email) {
            for (const message of parsedTranscript) {
                if (message.role === 'agent' &&
                    (message.text.includes("I've noted your email as") ||
                        message.text.includes("noted your email as"))) {
                    const emailMatch = message.text.match(/noted your email as\s+([^.]+)/i);
                    if (emailMatch && emailMatch[1]) {
                        // Try standard email format in the confirmation
                        const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/;
                        const match = emailMatch[1].match(emailRegex);
                        if (match) {
                            email = match[0];
                            console.log('Found standard email in agent confirmation:', email);
                            break;
                        }

                        // Try spoken email format in the confirmation
                        const spokenEmailRegex = /\b([a-zA-Z0-9._%+-]+)\s+at\s+([a-zA-Z0-9.-]+)\s+dot\s+([a-zA-Z]{2,})\b/i;
                        const spokenMatch = emailMatch[1].match(spokenEmailRegex);
                        if (spokenMatch) {
                            email = `${spokenMatch[1]}@${spokenMatch[2]}.${spokenMatch[3]}`;
                            console.log('Found spoken email in agent confirmation:', email);
                            break;
                        }
                    }
                }
            }
        }
    }

    // If still no email, try to extract from all messages
    if (!email) {
        email = extractEmailFromStringArray(allMessages) || null;
    }

    console.log('Final email determined for ticket:', email);

    // Extract questions from user messages
    const userQuestions = extractUserQuestions(conversation.transcript);
    console.log('Extracted user questions:', userQuestions);

    // Analyze conversation flow to determine resolution status
    let resolvedQuestions: string[] = [];
    let unresolvedQuestions: string[] = [];
    let hasPositiveAcknowledgment = false;
    let hasNegativeResponse = false;
    let topicKeywords: string[] = [];

    // If we have a parsed transcript and few messages in the conversation state,
    // use the parsed transcript for analysis instead
    if (parsedTranscript && parsedTranscript.length > 0 && conversation.transcript.length <= 4) {
        console.log('Using parsed transcript for conversation analysis');
        const transcriptFormatted = parsedTranscript.map(msg => ({
            role: msg.role === 'customer' ? 'user' : 'assistant',
            content: msg.text
        }));

        const analysisResult = await analyzeConversationResolution(transcriptFormatted, userQuestions);
        resolvedQuestions = analysisResult.resolvedQuestions;
        unresolvedQuestions = analysisResult.unresolvedQuestions;
        hasPositiveAcknowledgment = analysisResult.hasPositiveAcknowledgment;
        hasNegativeResponse = analysisResult.hasNegativeResponse;
        topicKeywords = analysisResult.topicKeywords;
    } else {
        // Use the conversation state transcript
        const analysisResult = await analyzeConversationResolution(conversation.transcript, userQuestions);
        resolvedQuestions = analysisResult.resolvedQuestions;
        unresolvedQuestions = analysisResult.unresolvedQuestions;
        hasPositiveAcknowledgment = analysisResult.hasPositiveAcknowledgment;
        hasNegativeResponse = analysisResult.hasNegativeResponse;
        topicKeywords = analysisResult.topicKeywords;
    }

    console.log('Resolution analysis:', {
        resolvedQuestions,
        unresolvedQuestions,
        hasPositiveAcknowledgment,
        hasNegativeResponse
    });

    // Check if AI couldn't find information in KB
    const noKBMatch = aiResponses.some(resp => checkForNoKBMatch(resp));

    // Determine if follow-up is required based on multiple factors
    const requiresFollowup =
        noKBMatch ||
        unresolvedQuestions.length > 0 ||
        hasNegativeResponse ||
        conversation.requiresHumanFollowup;

    // Determine priority based on resolution status and conversation analysis
    let priority: 'HIGH' | 'MEDIUM' | 'LOW' = 'MEDIUM';

    if (requiresFollowup) {
        if (unresolvedQuestions.length > 0 || noKBMatch) {
            priority = 'HIGH'; // High priority for unresolved questions or no KB match
        } else {
            priority = 'MEDIUM'; // Medium priority for other follow-up reasons
        }
    } else if (resolvedQuestions.length > 0 && unresolvedQuestions.length === 0 && hasPositiveAcknowledgment) {
        priority = 'LOW'; // Low priority for fully resolved conversations with positive acknowledgment
    }

    // Generate a descriptive subject using LLM
    const subject = await generateDescriptiveSubjectWithLLM(
        conversation.transcript,
        parsedTranscript
    );

    // Generate more detailed summary
    let summary = '';

    if (userQuestions.length > 0) {
        const questionSummary = userQuestions.length === 1
            ? `"${userQuestions[0]}"`
            : `multiple questions including "${userQuestions[0]}"`;

        summary += `The customer inquired about ${questionSummary}. `;
    } else {
        summary += `The customer contacted support about ${topicKeywords.join(', ') || 'Kayako services'}. `;
    }

    if (aiResponses.length > 0) {
        summary += `The AI provided assistance with ${generateKeyPoints(conversation.transcript)}. `;
    }

    if (resolvedQuestions.length > 0) {
        summary += `Successfully resolved ${resolvedQuestions.length} question(s). `;
    }

    return {
        subject,
        summary,
        email,
        requiresFollowup,
        priority,
        resolvedQuestions,
        unresolvedQuestions
    };
}
