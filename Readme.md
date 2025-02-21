# Speech Assistant with OpenAI Realtime API and Twilio

This project implements a voice assistant using OpenAI's Realtime API and Twilio for handling phone calls. The assistant can engage in real-time conversations, understand speech, and respond naturally.

## Prerequisites

- Node.js (v16 or higher)
- npm
- OpenAI API key
- Twilio account with credentials
- ngrok for local development

## Environment Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Copy `.env.example` to `.env` and fill in your credentials:
   ```
   OPENAI_API_KEY=your_openai_api_key
   TWILIO_ACCOUNT_SID=your_twilio_account_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   TWILIO_PHONE_NUMBER=your_twilio_phone_number
   PORT=5050
   ```

## Development

The project uses TypeScript and includes hot-reloading for development. To start development:

1. Build the TypeScript files:
   ```bash
   npm run build
   ```

2. In one terminal, start the ngrok tunnel:
   ```bash
   npm run tunnel
   ```
   Copy the ngrok URL and update your Twilio webhook settings to point to:
   - Voice Webhook: `{ngrok-url}/incoming-call`

3. In another terminal, start the development server with hot-reloading:
   ```bash
   npm run dev:server
   ```

## Project Structure

- `src/index.ts` - Main application file
- `src/server.ts` - Server setup and configuration
- `src/responseGenerator.ts` - Response generation logic
- `src/kayakoService.ts` - Kayako API integration

## Features

- Real-time voice conversations using OpenAI's Realtime API
- Twilio integration for phone call handling
- WebSocket-based media streaming
- TypeScript support
- Hot-reloading during development
- ngrok integration for local development

## Production

To run in production:

```bash
npm run prod
```

This will build the TypeScript files and start the server.

## Available Scripts

- `npm run build` - Build TypeScript files
- `npm run dev:server` - Start development server with hot-reloading
- `npm run tunnel` - Start ngrok tunnel for local development
- `npm run prod` - Build and start in production mode
- `npm start` - Start the server (without building)

## License

ISC
