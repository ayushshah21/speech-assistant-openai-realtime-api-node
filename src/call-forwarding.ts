import WebSocket from 'ws';
import pkg from 'twilio';
const { Twilio } = pkg;
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Types
export interface ConversationState {
    transcript: Array<{
        role: 'user' | 'assistant';
        content: string;
        timestamp: number;
        confidence?: number;
        level?: string;
    }>;
    rawUserInput: Array<{
        content: string;
        timestamp: number;
        confidence?: number;
        is_final?: boolean;
        duration?: number;
        start_time?: string;
        end_time?: string;
        level?: string;
    }>;
    userDetails: {
        email?: string;
        name?: string;
        phone?: string;
        hasProvidedEmail: boolean;
    };
    kbMatchFound: boolean;
    requiresHumanFollowup: boolean;
}

// Configuration
const TWILIO_ACCOUNT_SID = process.env.TWILIO_ACCOUNT_SID || '';
const TWILIO_AUTH_TOKEN = process.env.TWILIO_AUTH_TOKEN || '';
const SUPPORT_AGENT_NUMBER = process.env.SUPPORT_AGENT_NUMBER || '';
const ENABLE_CALL_FORWARDING = process.env.ENABLE_CALL_FORWARDING === 'true';
const FORWARDING_THRESHOLD = parseInt(process.env.FORWARDING_THRESHOLD || '3', 10);

// Initialize Twilio client
const twilioClient = new Twilio(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN);

/**
 * Determines if a call should be forwarded to a human agent based on conversation context
 * 
 * @param transcript - The conversation transcript
 * @param kbMatchFound - Whether a knowledge base match was found
 * @returns boolean indicating if the call should be forwarded
 */
export function shouldForwardCall(
    transcript: Array<{ role: string; content: string; timestamp?: number; confidence?: number; level?: string }>,
    kbMatchFound: boolean
): boolean {
    if (!ENABLE_CALL_FORWARDING || !SUPPORT_AGENT_NUMBER) {
        console.log('Call forwarding is disabled or support agent number not configured');
        return false;
    }

    // Check for explicit requests to speak with a human agent
    const lastUserMessages = transcript
        .filter(msg => msg.role === 'user')
        .slice(-3);

    const requestsAgent = lastUserMessages.some(msg => {
        const content = msg.content.toLowerCase();
        return content.includes('speak to agent') ||
            content.includes('talk to human') ||
            content.includes('real person') ||
            content.includes('speak with someone') ||
            content.includes('human agent') ||
            content.includes('transfer me') ||
            content.includes('connect me to') ||
            content.includes('speak to a human') ||
            content.includes('talk to a person') ||
            content.includes('speak with a representative') ||
            content.includes('connect me with someone') ||
            content.includes('need a human') ||
            content.includes('want to talk to a human') ||
            content.includes('agent please') ||
            content.includes('representative') ||
            content.includes('speak to support') ||
            content.includes('talk to support') ||
            content.includes('human support') ||
            content.includes('need help from a person') ||
            content.includes('can i speak to') ||
            content.includes('can i talk to');
    });

    // Check for repeated failures or confusion
    const repeatedFailures = checkForRepeatedFailures(transcript);

    // Check for complex issue types
    const complexIssueDetected = detectComplexIssue(transcript);

    // Log the decision factors
    console.log('Call forwarding decision factors:', {
        requestsAgent,
        kbMatchFound,
        repeatedFailures,
        complexIssueDetected
    });

    return requestsAgent || !kbMatchFound || repeatedFailures || complexIssueDetected;
}

/**
 * Check if there are repeated failures in the conversation
 * 
 * @param transcript - The conversation transcript
 * @returns boolean indicating if there are repeated failures
 */
function checkForRepeatedFailures(transcript: Array<{ role: string; content: string; timestamp?: number; confidence?: number; level?: string }>): boolean {
    // Count negative responses from the user
    let negativeResponseCount = 0;
    let consecutiveNegativeResponses = 0;

    // Patterns indicating user frustration or confusion
    const negativePatterns = [
        'not what i',
        'doesn\'t answer',
        'didn\'t answer',
        'not helpful',
        'don\'t understand',
        'not working',
        'incorrect',
        'wrong',
        'no that\'s not',
        'that doesn\'t help'
    ];

    // Analyze user messages for negative patterns
    for (let i = 0; i < transcript.length; i++) {
        const entry = transcript[i];
        if (entry.role === 'user') {
            const content = entry.content.toLowerCase();

            const isNegative = negativePatterns.some(pattern => content.includes(pattern));

            if (isNegative) {
                negativeResponseCount++;
                consecutiveNegativeResponses++;

                // If we have multiple consecutive negative responses, that's a strong signal
                if (consecutiveNegativeResponses >= 2) {
                    return true;
                }
            } else {
                consecutiveNegativeResponses = 0;
            }
        }
    }

    // If total negative responses exceed our threshold, forward the call
    return negativeResponseCount >= FORWARDING_THRESHOLD;
}

/**
 * Detect if the conversation involves a complex issue that requires human intervention
 * 
 * @param transcript - The conversation transcript
 * @returns boolean indicating if a complex issue is detected
 */
function detectComplexIssue(transcript: Array<{ role: string; content: string; timestamp?: number; confidence?: number; level?: string }>): boolean {
    // Keywords indicating complex issues
    const complexIssueKeywords = [
        'custom integration',
        'api',
        'billing',
        'refund',
        'cancel',
        'subscription',
        'legal',
        'gdpr',
        'data protection',
        'security breach',
        'urgent',
        'emergency',
        'critical',
        'broken',
        'not working at all'
    ];

    // Check user messages for complex issue keywords
    for (const entry of transcript) {
        if (entry.role === 'user') {
            const content = entry.content.toLowerCase();
            if (complexIssueKeywords.some(keyword => content.includes(keyword))) {
                return true;
            }
        }
    }

    return false;
}

/**
 * Transfer an ongoing call to a human agent
 * 
 * @param connection - The WebSocket connection
 * @param callSid - The Twilio Call SID
 * @param streamSid - The Twilio Stream SID
 * @param conversationState - The current conversation state
 * @param addToTranscript - Function to add messages to the transcript
 */
export async function transferCallToAgent(
    connection: WebSocket,
    callSid: string,
    streamSid: string,
    conversationState: ConversationState,
    addToTranscript: (role: 'user' | 'assistant', content: string, confidence?: number) => void
): Promise<void> {
    try {
        console.log(`Initiating call transfer process:`);
        console.log(`- Call SID: ${callSid}`);
        console.log(`- Stream SID: ${streamSid}`);
        console.log(`- Support Agent Number: ${SUPPORT_AGENT_NUMBER}`);
        console.log(`- WebSocket State: ${connection.readyState === WebSocket.OPEN ? 'OPEN' : 'NOT_OPEN'}`);

        // 1. Inform the user about the transfer
        const transferMessage = "I'll connect you with a support specialist who can better assist you with this issue. Please hold while I transfer your call.";

        // 2. Add the transfer message to the transcript
        addToTranscript('assistant', transferMessage);
        console.log('Added transfer message to transcript');

        // 3. Generate a summary of the conversation for the agent
        const conversationSummary = generateConversationSummary(conversationState);
        console.log('Generated conversation summary for agent');

        // 4. Execute the transfer via Twilio API
        console.log('Executing call transfer via Twilio API...');

        try {
            await executeCallTransfer(callSid, conversationSummary);
            console.log(`Call ${callSid} successfully transferred to agent at ${SUPPORT_AGENT_NUMBER}`);
        } catch (transferError) {
            console.error('Primary transfer method failed:', transferError);

            // If the call transfer fails, try to get a new Call SID and retry
            console.log('Attempting to get a fresh Call SID and retry transfer...');
            const freshCallSid = await getCallSidFromStreamSid(streamSid);

            if (freshCallSid && freshCallSid !== callSid) {
                console.log(`Got fresh Call SID: ${freshCallSid}, retrying transfer`);
                await executeCallTransfer(freshCallSid, conversationSummary);
                console.log(`Call ${freshCallSid} successfully transferred to agent at ${SUPPORT_AGENT_NUMBER}`);
            } else {
                throw transferError; // Re-throw if we couldn't get a new SID or it's the same
            }
        }

    } catch (error) {
        console.error('Error transferring call:', error);

        // Log detailed error information
        if (error instanceof Error) {
            console.error('Error message:', error.message);
            console.error('Error stack:', error.stack);
        }

        // Check if the error is related to an invalid Call SID
        const errorMessage = error instanceof Error ? error.message : String(error);
        const isInvalidCallSid = errorMessage.includes('invalid CallSid') ||
            errorMessage.includes('not found') ||
            errorMessage.includes('does not exist');

        // Send an appropriate error message to the user
        let userErrorMessage = "I'm sorry, I'm having trouble connecting you with a specialist. ";

        if (isInvalidCallSid) {
            userErrorMessage += "There seems to be an issue with the call connection. ";
        }

        userErrorMessage += `Please try calling our support line directly at ${SUPPORT_AGENT_NUMBER}`;

        // Add the error message to the transcript
        if (connection.readyState === WebSocket.OPEN) {
            addToTranscript('assistant', userErrorMessage);
            console.log('Added error message to transcript');
        }
    }
}

/**
 * Generate a summary of the conversation for the agent
 * 
 * @param conversationState - The current conversation state
 * @returns A summary of the conversation
 */
function generateConversationSummary(conversationState: ConversationState): string {
    const userEmail = conversationState.userDetails.email || 'Not provided';
    const userPhone = conversationState.userDetails.phone || 'Not provided';

    // Extract the last few messages for context
    const recentMessages = conversationState.transcript
        .slice(-5)
        .map(msg => `${msg.role === 'user' ? 'Customer' : 'AI'}: ${msg.content}`)
        .join('\n');

    return `
Call transferred from AI assistant.
Customer Email: ${userEmail}
Customer Phone: ${userPhone}
Recent Conversation:
${recentMessages}
`;
}

/**
 * Execute the call transfer using Twilio's API
 * 
 * @param callSid - The Twilio Call SID
 * @param conversationSummary - A summary of the conversation
 */
async function executeCallTransfer(callSid: string, conversationSummary: string): Promise<void> {
    try {
        console.log(`Attempting to transfer call ${callSid} to ${SUPPORT_AGENT_NUMBER}`);

        // Validate the Call SID format
        if (!callSid.startsWith('CA')) {
            console.warn(`Call SID ${callSid} doesn't match expected format (should start with CA)`);
        }

        // First, verify the call exists and is active
        try {
            const callInstance = await twilioClient.calls(callSid).fetch();
            console.log(`Call status: ${callInstance.status}`);

            // Check if the call is in a state that can be modified
            if (callInstance.status !== 'in-progress' && callInstance.status !== 'ringing') {
                console.warn(`Call ${callSid} is in ${callInstance.status} state, which may not support transfer`);
            }
        } catch (fetchError) {
            console.error(`Error fetching call ${callSid}:`, fetchError);
            throw new Error(`Call ${callSid} not found or not accessible`);
        }

        // Create TwiML for the transfer
        // Using <Dial> with <Number> for a simple transfer
        const twiml = `
<Response>
    <Say>Please hold while I connect you with a support specialist.</Say>
    <Dial callerId="${process.env.TWILIO_PHONE_NUMBER}">
        <Number>${SUPPORT_AGENT_NUMBER}</Number>
    </Dial>
    <Say>I'm sorry, but our support team is unavailable at the moment. Please try calling back during business hours.</Say>
</Response>`;

        // Log the TwiML for debugging
        console.log('Transfer TwiML:', twiml);

        // Update the call with the new TwiML
        // This redirects the current call to follow the new TwiML instructions
        const result = await twilioClient.calls(callSid).update({
            twiml: twiml
        });

        console.log(`Transfer initiated for call ${callSid}. New status: ${result.status}`);
        console.log('Conversation summary for agent:', conversationSummary);

    } catch (error) {
        console.error('Error executing call transfer:', error);
        // Log more details about the error
        if (error instanceof Error) {
            console.error('Error message:', error.message);
            console.error('Error stack:', error.stack);
        }

        // Try an alternative approach if the first method fails
        try {
            console.log('Attempting alternative transfer method...');

            // Create a new outbound call to the agent and connect it to the current call
            const newCall = await twilioClient.calls.create({
                to: SUPPORT_AGENT_NUMBER,
                from: process.env.TWILIO_PHONE_NUMBER || '',
                twiml: `<Response><Say>Connecting you to a customer who needs assistance.</Say></Response>`
            });

            console.log(`Created new outbound call to agent: ${newCall.sid}`);

        } catch (altError) {
            console.error('Alternative transfer method also failed:', altError);
            throw error; // Re-throw the original error
        }
    }
}

/**
 * Get the Call SID from a Stream SID using Twilio's API
 * 
 * @param streamSid - The Twilio Stream SID
 * @returns The Call SID or null if not found
 */
export async function getCallSidFromStreamSid(streamSid: string): Promise<string | null> {
    try {
        console.log(`Getting Call SID for Stream SID: ${streamSid}`);

        // The Media Stream SID doesn't directly map to a Call SID
        // We need to query active calls and check their properties

        // Get all active calls
        const calls = await twilioClient.calls.list({ status: 'in-progress' });
        console.log(`Found ${calls.length} active calls`);

        if (calls.length === 0) {
            console.log('No active calls found. The call might have ended or not started yet.');
            return null;
        }

        // If there's only one active call, it's likely the one we want
        if (calls.length === 1) {
            console.log(`Only one active call found with SID: ${calls[0].sid}`);
            return calls[0].sid;
        }

        // If there are multiple calls, log them for debugging
        calls.forEach(call => {
            console.log(`Active call: ${call.sid}, From: ${call.from}, To: ${call.to}, Status: ${call.status}`);
        });

        // For now, return the most recent call as a fallback
        // Sort calls by dateCreated in descending order
        const sortedCalls = calls.sort((a, b) => {
            return new Date(b.dateCreated).getTime() - new Date(a.dateCreated).getTime();
        });

        console.log(`Using most recent call with SID: ${sortedCalls[0].sid}`);
        return sortedCalls[0].sid;

    } catch (error) {
        console.error('Error getting Call SID from Stream SID:', error);

        // Log detailed error information
        if (error instanceof Error) {
            console.error('Error message:', error.message);
            console.error('Error stack:', error.stack);
        }

        return null;
    }
} 