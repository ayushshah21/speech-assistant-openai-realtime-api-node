import WebSocket from 'ws';
import pkg from 'twilio';
const { Twilio } = pkg;
import dotenv from 'dotenv';
import { OpenAI } from 'openai';

// Load environment variables
dotenv.config();

// Initialize OpenAI with API key from environment variables
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

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
 * Evaluate if a call should be forwarded to a human agent using LLM
 * 
 * @param transcript - The conversation transcript
 * @returns Promise with decision object containing shouldForward boolean and reason string
 */
export async function evaluateForwardingWithLLM(
    transcript: Array<{ role: string; content: string; timestamp?: number; confidence?: number; level?: string }>
): Promise<{
    shouldForward: boolean;
    reason: string;
}> {
    try {
        // Get the last AI response if available
        const aiResponses = transcript
            .filter(msg => msg.role === 'assistant')
            .map(msg => msg.content);

        // If there are no AI responses yet, don't forward
        if (aiResponses.length === 0) {
            return {
                shouldForward: false,
                reason: "No AI responses yet to evaluate"
            };
        }

        // Get the most recent AI response
        const lastAIResponse = aiResponses[aiResponses.length - 1];

        // Direct indicators that the AI is suggesting forwarding to a human
        const forwardingPhrases = [
            "connect you with a support specialist",
            "transfer you to a human agent",
            "connect you with a human",
            "transfer your call",
            "speak with a representative",
            "connect you with an agent",
            "transfer you to a specialist",
            "put you in touch with our support team",
            "escalate this to our support team",
            "have a support agent contact you",
            "I'll connect you with a support",
            "I'll transfer you to",
            "let me transfer you",
            "I'll get a human agent",
            "I'll have someone assist you",
            "support specialist follow up with you",
            "have a support specialist follow up",
            "can't forward you directly",
            "prefer to speak with a human agent",
            "would like to speak with a human",
            "understand you'd like to speak with a human"
        ];

        // Check if the AI's response directly indicates forwarding
        const directForwardingIndicated = forwardingPhrases.some(phrase =>
            lastAIResponse.toLowerCase().includes(phrase.toLowerCase())
        );

        if (directForwardingIndicated) {
            return {
                shouldForward: true,
                reason: "AI response indicates forwarding is appropriate: " + lastAIResponse.substring(0, 100) + "..."
            };
        }

        // Check if the AI is indicating it can't help
        const cannotHelpPhrases = [
            "I don't have that information",
            "I don't have access to",
            "I'm unable to assist with",
            "I can't access your account",
            "I don't have the ability to",
            "that's beyond my capabilities",
            "I'm not able to help with",
            "would require human assistance",
            "I don't have specific information",
            "I don't have enough information",
            "I cannot access your personal",
            "I don't have access to your specific",
            "I cannot view your account",
            "I'm not able to see your",
            "I cannot perform that action",
            "I'm limited in what I can do",
            "I cannot make changes to your account",
            "I don't have the authorization to"
        ];

        const aiCannotHelp = cannotHelpPhrases.some(phrase =>
            lastAIResponse.toLowerCase().includes(phrase.toLowerCase())
        );

        // Phrases that should NOT trigger forwarding (routine conversation)
        const routineConversationPhrases = [
            "provide your email",
            "may I have your email",
            "could you share your email",
            "what's your email",
            "what is your email",
            "could I get your email",
            "can I get your email",
            "follow up if needed",
            "for our records",
            "to keep you updated",
            "to send you updates",
            "to contact you later",
            "to send you a confirmation",
            "is there anything else",
            "can I help you with anything else",
            "is there something else",
            "would you like me to help with anything else",
            "do you have any other questions",
            "is there anything else you'd like to know"
        ];

        // Check if this is just a routine email collection or conversation closing
        const isRoutineConversation = routineConversationPhrases.some(phrase =>
            lastAIResponse.toLowerCase().includes(phrase.toLowerCase())
        );

        // If this is just asking for an email or closing the conversation as part of normal flow, don't forward
        if (isRoutineConversation && !directForwardingIndicated && !aiCannotHelp) {
            return {
                shouldForward: false,
                reason: "AI is engaging in routine conversation (collecting information or closing)"
            };
        }

        // Knowledge base indicators - phrases that suggest the AI is using the knowledge base
        const kbIndicatorPhrases = [
            "according to our knowledge base",
            "based on our information",
            "our documentation states",
            "our records show",
            "our knowledge base indicates",
            "according to our documentation",
            "our support documentation",
            "our help center",
            "our support articles",
            "our guide explains",
            "our FAQ states",
            "as mentioned in our documentation",
            "as outlined in our guide",
            "as described in our help center"
        ];

        const isUsingKnowledgeBase = kbIndicatorPhrases.some(phrase =>
            lastAIResponse.toLowerCase().includes(phrase.toLowerCase())
        );

        // If the AI is clearly using the knowledge base, don't forward
        if (isUsingKnowledgeBase && !directForwardingIndicated && !aiCannotHelp) {
            return {
                shouldForward: false,
                reason: "AI is providing information from the knowledge base"
            };
        }

        if (aiCannotHelp) {
            return {
                shouldForward: true,
                reason: "AI indicates it cannot help with this request: " + lastAIResponse.substring(0, 100) + "..."
            };
        }

        // Check if the AI is providing a substantive answer
        // If the response is longer and doesn't contain forwarding or inability phrases,
        // it's likely providing a real answer
        const isSubstantiveAnswer = lastAIResponse.length > 100 &&
            !directForwardingIndicated &&
            !aiCannotHelp;

        if (isSubstantiveAnswer) {
            return {
                shouldForward: false,
                reason: "AI is providing a substantive answer to the user's question"
            };
        }

        // If we're not sure, use the LLM to evaluate
        const formattedTranscript = transcript
            .map(msg => `${msg.role.toUpperCase()}: ${msg.content}`)
            .join('\n');

        // Create a focused prompt for evaluating the AI's response
        const prompt = `
You are evaluating an AI assistant's response to determine if a call should be forwarded to a human agent.

Conversation transcript:
${formattedTranscript}

Focus EXCLUSIVELY on the AI ASSISTANT's responses, not the user's questions. Determine if:
1. The AI is clearly indicating it will transfer the call or connect the user with a human
2. The AI is stating it cannot help with the request or lacks necessary information
3. The AI is providing a substantive, helpful answer to the user's question

IMPORTANT GUIDELINES:
- If the AI's response mentions transferring, connecting to a specialist, or similar phrases, recommend forwarding
- If the AI indicates it cannot help or lacks information, recommend forwarding
- If the AI is providing a real answer that addresses the user's question, do NOT recommend forwarding
- If the AI is simply asking for contact information (email, name, etc.) as part of normal conversation, do NOT recommend forwarding
- If the AI is asking "is there anything else I can help with" or similar closing phrases, do NOT recommend forwarding
- If the AI is providing information from a knowledge base (mentions documentation, guides, FAQs, etc.), do NOT recommend forwarding
- If the AI is helping with password reset instructions or other common support tasks, do NOT recommend forwarding
- Asking for an email "to follow up if needed" is normal conversation flow and should NOT trigger forwarding
- Ignore the user's request content - focus ONLY on what the AI has responded with

CRITICAL: Be conservative about recommending forwarding. Only recommend forwarding if the AI is CLEARLY indicating it cannot help or is explicitly suggesting a transfer.

Respond with JSON only in this format:
{"shouldForward": boolean, "reason": "brief explanation focusing on the AI's response"}
`;

        // Call OpenAI to evaluate the conversation
        const openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY,
        });

        const response = await openai.chat.completions.create({
            model: "gpt-4-turbo",
            messages: [
                { role: "system", content: prompt }
            ],
            temperature: 0.1,
            max_tokens: 150,
            response_format: { type: "json_object" }
        });

        const responseText = response.choices[0].message.content || '';
        console.log('LLM evaluation response:', responseText);

        try {
            // Parse the JSON response
            const result = JSON.parse(responseText);
            return {
                shouldForward: result.shouldForward,
                reason: result.reason
            };
        } catch (error) {
            console.error('Error parsing LLM response:', error);
            // Default to not forwarding if we can't parse the response
            return {
                shouldForward: false,
                reason: "Error evaluating forwarding decision"
            };
        }
    } catch (error) {
        console.error('Error in evaluateForwardingWithLLM:', error);
        // Default to not forwarding on error
        return {
            shouldForward: false,
            reason: "Error in forwarding evaluation"
        };
    }
}

/**
 * Transfer an ongoing call to a human agent
 * 
 * @param connection - The WebSocket connection
 * @param callSid - The Twilio Call SID
 * @param streamSid - The Twilio Stream SID
 * @param conversationState - The current conversation state
 * @param addToTranscript - Function to add messages to the transcript
 * @param isSpeaking - Optional flag indicating if the AI is currently speaking
 * @param isResponseFullyDone - Optional flag indicating if the AI's response is complete
 * @param skipAcknowledgment - Optional flag to skip sending the acknowledgment message
 */
export async function transferCallToAgent(
    connection: WebSocket,
    callSid: string,
    streamSid: string,
    conversationState: ConversationState,
    addToTranscript: (role: 'user' | 'assistant', content: string, confidence?: number) => void,
    isSpeaking?: boolean,
    isResponseFullyDone?: boolean,
    skipAcknowledgment?: boolean
): Promise<void> {
    try {
        console.log(`Initiating call transfer process:`);
        console.log(`- Call SID: ${callSid}`);
        console.log(`- Stream SID: ${streamSid}`);
        console.log(`- Support Agent Number: ${SUPPORT_AGENT_NUMBER}`);
        console.log(`- WebSocket State: ${connection.readyState === WebSocket.OPEN ? 'OPEN' : 'NOT_OPEN'}`);
        console.log(`- AI Speaking: ${isSpeaking ? 'Yes' : 'No'}`);
        console.log(`- Response Fully Done: ${isResponseFullyDone ? 'Yes' : 'No'}`);
        console.log(`- Skip Acknowledgment: ${skipAcknowledgment ? 'Yes' : 'No'}`);

        // If the AI is currently speaking, wait for a short time to allow it to finish
        // This helps prevent interrupting the AI mid-sentence
        if (isSpeaking === true) {
            console.log('AI is currently speaking. Waiting briefly before initiating transfer...');
            // Wait for a short time (1.5 seconds) to allow the AI to finish its current sentence
            await new Promise(resolve => setTimeout(resolve, 1500));
            console.log('Wait complete, proceeding with transfer');
        }

        // Only send the transfer message if we're not skipping the acknowledgment
        if (!skipAcknowledgment) {
            // 1. Inform the user about the transfer with a clear, complete message
            const transferMessage = "I understand you'd like to speak with a human agent. I'll connect you with a support specialist who can better assist you with this issue. Please hold while I transfer your call.";

            // 2. Add the transfer message to the transcript
            addToTranscript('assistant', transferMessage);
            console.log('Added transfer message to transcript');
        } else {
            console.log('Skipping acknowledgment message as AI has already acknowledged the request');

            // Add just the transfer part to the transcript
            const transferOnlyMessage = "I'll connect you with a support specialist who can better assist you with this issue. Please hold while I transfer your call.";
            addToTranscript('assistant', transferOnlyMessage);
            console.log('Added transfer-only message to transcript');
        }

        // 3. Generate a summary of the conversation for the agent
        const conversationSummary = generateConversationSummary(conversationState);
        console.log('Generated conversation summary for agent');

        // 4. Add a natural pause before executing the transfer
        // This creates a more natural conversational flow
        console.log('Adding a brief pause before executing transfer...');
        await new Promise(resolve => setTimeout(resolve, 2000));

        // 5. Execute the transfer via Twilio API
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