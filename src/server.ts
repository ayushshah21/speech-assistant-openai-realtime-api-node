import dotenv from 'dotenv';
import path from 'path';

// Load environment variables
dotenv.config({
    path: path.resolve(process.cwd(), '.env')
});

import express from 'express';
import bodyParser from 'body-parser';
import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import { IncomingMessage } from 'http';
import { handleIncomingCall, handleMediaStream, endCall } from '../telephony/twilioHandler';

const app = express();
const server = createServer(app);

// Create a WS server listening at path "/media"
const wss = new WebSocketServer({ noServer: true });

const port = process.env.PORT || 3000;

// Logging environment variables
console.log('Environment variables loaded:', {
    NODE_ENV: process.env.NODE_ENV,
    ELEVEN_LABS_API_KEY: process.env.ELEVEN_LABS_API_KEY ? 'Set' : 'Not set',
    PORT: process.env.PORT
});

// Middleware
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

// Health check endpoint
app.get('/health', (req, res) => {
    res.status(200).json({ status: 'healthy' });
});

// Twilio webhook endpoints
app.post('/voice', handleIncomingCall);
app.post('/voice/status', (req, res) => {
    const { CallSid, CallStatus } = req.body;
    if (CallStatus === 'completed' || CallStatus === 'failed') {
        endCall(CallSid);
    }
    res.sendStatus(200);
});

// Upgrade HTTP => WS for "/media/..."
server.on('upgrade', (request: IncomingMessage, socket, head) => {
    // e.g. GET /media/CA7e052eee89d6e3c2...
    const url = new URL(request.url || '', `http://${request.headers.host}`);
    if (url.pathname.startsWith('/media/')) {
        wss.handleUpgrade(request, socket, head, (ws) => {
            // Extract the callSid from the path
            // e.g. /media/CA7e052eee => "CA7e052eee"
            const pathParts = url.pathname.split('/');
            const callSid = pathParts[2] || 'unknownSid';

            console.log('WebSocket connected for callSid:', callSid);
            handleMediaStream(ws, callSid);
        });
    } else {
        socket.destroy();
    }
});

// Start server
server.listen(port, () => {
    console.log(`Server running on port ${port}`);
    console.log(`Twilio configuration => Account SID: ${process.env.TWILIO_ACCOUNT_SID}`);
});
