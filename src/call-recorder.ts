import { Buffer } from 'buffer';
import { Lame } from 'node-lame';
import * as fs from 'fs';
import * as path from 'path';

// G711 ulaw to linear PCM conversion table
const ULAW_TO_LINEAR = new Int16Array(256);
for (let i = 0; i < 256; i++) {
    const ulaw = ~i;
    let t = ((ulaw & 0x0F) << 3) + 0x84;
    t <<= ((ulaw & 0x70) >> 4);
    ULAW_TO_LINEAR[i] = (ulaw & 0x80) ? (0x84 - t) : (t - 0x84);
}

// Metadata about the recording
interface RecordingMetadata {
    streamSid: string;
    startTime: number;
    duration?: number;
    fileSize?: number;
    mp3Path?: string;
}

// Structure for storing audio chunks
interface AudioChunk {
    timestamp: number;
    payload: Buffer;
    isAI: boolean;
}

/**
 * CallRecorder handles recording and storing both user and AI audio during a call.
 */
export class CallRecorder {
    private audioChunks: AudioChunk[];
    private metadata: RecordingMetadata;
    private recordingsDir: string;

    constructor(streamSid: string) {
        this.audioChunks = [];
        this.metadata = {
            streamSid,
            startTime: Date.now()
        };

        // Ensure recordings directory exists
        this.recordingsDir = path.join(process.cwd(), 'recordings');
        if (!fs.existsSync(this.recordingsDir)) {
            fs.mkdirSync(this.recordingsDir, { recursive: true });
        }
    }

    /**
     * Convert G711 ulaw buffer to PCM buffer
     */
    private ulawToPcm(ulawData: Buffer): Buffer {
        const pcmData = Buffer.alloc(ulawData.length * 2);
        for (let i = 0; i < ulawData.length; i++) {
            pcmData.writeInt16LE(ULAW_TO_LINEAR[ulawData[i]], i * 2);
        }
        return pcmData;
    }

    /**
     * Create WAV header for PCM audio
     */
    private createWavHeader(dataLength: number): Buffer {
        const buffer = Buffer.alloc(44);
        const numChannels = 1;
        const sampleRate = 8000;
        const bitsPerSample = 16;
        const byteRate = sampleRate * numChannels * (bitsPerSample / 8);

        // RIFF header
        buffer.write('RIFF', 0);
        buffer.writeUInt32LE(dataLength + 36, 4);
        buffer.write('WAVE', 8);

        // fmt subchunk
        buffer.write('fmt ', 12);
        buffer.writeUInt32LE(16, 16);                          // Subchunk1Size
        buffer.writeUInt16LE(1, 20);                           // AudioFormat (1 = PCM)
        buffer.writeUInt16LE(numChannels, 22);                 // NumChannels
        buffer.writeUInt32LE(sampleRate, 24);                  // SampleRate
        buffer.writeUInt32LE(byteRate, 28);                    // ByteRate
        buffer.writeUInt16LE(numChannels * (bitsPerSample / 8), 32); // BlockAlign
        buffer.writeUInt16LE(bitsPerSample, 34);              // BitsPerSample

        // data subchunk
        buffer.write('data', 36);
        buffer.writeUInt32LE(dataLength, 40);

        return buffer;
    }

    /**
     * Add user audio chunk to the recording
     */
    addUserAudio(payload: string): void {
        this.addAudioChunk(payload, false);
    }

    /**
     * Add AI audio chunk to the recording
     */
    addAIAudio(payload: string): void {
        this.addAudioChunk(payload, true);
    }

    /**
     * Internal method to add audio chunks
     */
    private addAudioChunk(payload: string, isAI: boolean): void {
        const chunk: AudioChunk = {
            timestamp: Date.now(),
            payload: Buffer.from(payload, 'base64'),
            isAI
        };
        this.audioChunks.push(chunk);
    }

    /**
     * Get the current duration of the recording in seconds
     */
    getDuration(): number {
        if (this.audioChunks.length === 0) return 0;
        const firstChunk = this.audioChunks[0];
        const lastChunk = this.audioChunks[this.audioChunks.length - 1];
        return (lastChunk.timestamp - firstChunk.timestamp) / 1000;
    }

    /**
     * Finish recording and convert to MP3
     * Returns the path to the MP3 file
     */
    async finishRecording(): Promise<string> {
        if (this.audioChunks.length === 0) {
            throw new Error('No audio recorded');
        }

        try {
            // Sort chunks by timestamp
            this.audioChunks.sort((a, b) => a.timestamp - b.timestamp);

            // Convert all chunks to PCM
            const pcmChunks = this.audioChunks.map(chunk => this.ulawToPcm(chunk.payload));

            // Combine all PCM chunks
            const pcmData = Buffer.concat(pcmChunks);

            // Create WAV file with proper header
            const wavHeader = this.createWavHeader(pcmData.length);
            const wavPath = path.join(this.recordingsDir, `${this.metadata.streamSid}.wav`);

            // Write WAV file with header and PCM audio data
            fs.writeFileSync(wavPath, Buffer.concat([wavHeader, pcmData]));

            // Convert to MP3 with optimized settings
            const mp3Path = path.join(this.recordingsDir, `${this.metadata.streamSid}.mp3`);
            const encoder = new Lame({
                output: mp3Path,
                bitrate: 128,
                raw: false, // Not raw anymore since we're using WAV
                sfreq: 8, // 8kHz
                signed: true,
                'little-endian': true,
                bitwidth: 16,
                quality: 5, // Higher quality setting
                mode: 'm' // Mono
            }).setFile(wavPath);

            await encoder.encode();

            // Clean up temporary WAV file
            fs.unlinkSync(wavPath);

            // Update metadata
            this.metadata.duration = this.getDuration();
            this.metadata.mp3Path = mp3Path;
            const stats = fs.statSync(mp3Path);
            this.metadata.fileSize = stats.size;

            return mp3Path;
        } catch (error) {
            console.error('Error converting audio to MP3:', error);
            throw error;
        }
    }

    /**
     * Get recording metadata
     */
    getMetadata(): RecordingMetadata {
        return { ...this.metadata };
    }
}

// Export a factory function to create new recorder instances
export function createCallRecorder(streamSid: string): CallRecorder {
    return new CallRecorder(streamSid);
}