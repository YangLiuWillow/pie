/**
 * End-to-end smoke test for the Pie ↔ OpenClaw integration.
 *
 * Boots a real `pie serve` (dummy driver, random tokens), installs the
 * `openclaw-chat` inferlet, and verifies that:
 *
 *   1. createPieStream() connects and launches a process.
 *   2. The inferlet receives messages/tools, generates tokens, and streams
 *      JSON-line events back.
 *   3. The StreamFunction yields the correct OpenClaw event sequence:
 *      start → text_delta* → done.
 *   4. The final AssistantMessage has the expected shape.
 *
 * Gated on environment variables:
 *
 *   PIE_BIN    — path to the `pie` binary
 *   PIE_WASM   — path to openclaw_chat.wasm
 *   PIE_MANIFEST — path to openclaw-chat/Pie.toml
 *
 * Run with:
 *
 *   PIE_BIN=../../target/release/pie \
 *   PIE_WASM=../../inferlets/openclaw-chat/target/wasm32-wasip2/release/openclaw_chat.wasm \
 *   PIE_MANIFEST=../../inferlets/openclaw-chat/Pie.toml \
 *   node --test tests/test_e2e_smoke.mjs
 */

import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createConnection } from 'node:net';
import { setTimeout as sleep } from 'node:timers/promises';
import { PieClient } from '@pie-project/client';
import { createPieStream } from '../src/stream.ts';

const PIE_BIN = process.env.PIE_BIN;
const PIE_WASM = process.env.PIE_WASM;
const PIE_MANIFEST = process.env.PIE_MANIFEST;
const PIE_CONFIG = process.env.PIE_CONFIG || new URL('fixtures/pie_dummy_config.toml', import.meta.url).pathname;

const INFERLET_NAME = 'openclaw-chat@0.1.0';

async function waitForPort(host, port, timeoutMs = 120_000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        try {
            await new Promise((resolve, reject) => {
                const sock = createConnection({ host, port }, () => {
                    sock.destroy();
                    resolve();
                });
                sock.on('error', reject);
                sock.setTimeout(1000, () => { sock.destroy(); reject(new Error('timeout')); });
            });
            return;
        } catch {
            await sleep(500);
        }
    }
    throw new Error(`pie serve never bound to ${host}:${port}`);
}

describe('OpenClaw ↔ Pie E2E', { skip: !PIE_BIN || !PIE_WASM || !PIE_MANIFEST }, () => {
    let pieProc;
    let pieUri;
    let port;

    before(async () => {
        // Find a free port.
        const net = await import('node:net');
        port = await new Promise((resolve, reject) => {
            const srv = net.createServer();
            srv.listen(0, '127.0.0.1', () => {
                const p = srv.address().port;
                srv.close(() => resolve(p));
            });
            srv.on('error', reject);
        });

        // Boot pie serve.
        pieProc = spawn(PIE_BIN, [
            'serve',
            '--config', PIE_CONFIG,
            '--port', String(port),
            '--no-auth',
        ], { stdio: ['ignore', 'pipe', 'pipe'] });

        pieProc.stdout.on('data', (d) => process.stderr.write(`[pie] ${d}`));
        pieProc.stderr.on('data', (d) => process.stderr.write(`[pie] ${d}`));

        await waitForPort('127.0.0.1', port);
        pieUri = `ws://127.0.0.1:${port}`;

        // Install the inferlet.
        const client = new PieClient(pieUri);
        await client.connect();
        await client.installProgram(PIE_WASM, PIE_MANIFEST, true);
        await client.close();
    });

    after(() => {
        if (pieProc) {
            pieProc.kill('SIGTERM');
            try { pieProc.kill('SIGKILL'); } catch {}
        }
    });

    it('streams a simple chat completion', async () => {
        const stream = createPieStream({ pieUri, inferlet: INFERLET_NAME });

        const context = {
            systemPrompt: 'You are a helpful assistant.',
            messages: [
                { role: 'user', content: 'Hello, how are you?' },
            ],
            tools: [],
        };

        const events = [];
        for await (const event of stream(null, context, { maxTokens: 32 })) {
            events.push(event);
        }

        // Must have: start, at least one text_delta, done.
        assert.ok(events.length >= 3, `Expected >= 3 events, got ${events.length}`);
        assert.equal(events[0].type, 'start');
        assert.equal(events[events.length - 1].type, 'done');

        const textDeltas = events.filter((e) => e.type === 'text_delta');
        assert.ok(textDeltas.length > 0, 'Expected at least one text_delta');

        const done = events[events.length - 1];
        assert.ok(done.message, 'done event must have a message');
        assert.equal(done.message.role, 'assistant');
        assert.ok(done.message.content.length > 0, 'assistant must have content');
        assert.ok(done.message.usage.outputTokens > 0, 'must have generated tokens');
    });

    it('handles tool-equipped conversations', async () => {
        const stream = createPieStream({ pieUri, inferlet: INFERLET_NAME });

        const context = {
            systemPrompt: 'You are a helpful assistant.',
            messages: [
                { role: 'user', content: 'What is the weather in Paris?' },
            ],
            tools: [
                {
                    type: 'function',
                    function: {
                        name: 'get_weather',
                        description: 'Get current weather for a location',
                        parameters: {
                            type: 'object',
                            properties: {
                                location: { type: 'string', description: 'City name' },
                            },
                            required: ['location'],
                        },
                    },
                },
            ],
        };

        const events = [];
        for await (const event of stream(null, context, { maxTokens: 64 })) {
            events.push(event);
        }

        assert.equal(events[0].type, 'start');
        assert.equal(events[events.length - 1].type, 'done');

        // With dummy driver (random tokens), we can't guarantee a tool call
        // will be produced, but the inferlet must not crash. The done event
        // must have a valid stop reason.
        const done = events[events.length - 1];
        assert.ok(
            ['stop', 'length', 'toolUse'].includes(done.reason),
            `unexpected stop reason: ${done.reason}`,
        );
    });

    it('respects max_tokens limit', async () => {
        const stream = createPieStream({ pieUri, inferlet: INFERLET_NAME });

        const context = {
            messages: [
                { role: 'user', content: 'Count from 1 to 1000.' },
            ],
        };

        const events = [];
        for await (const event of stream(null, context, { maxTokens: 8 })) {
            events.push(event);
        }

        const done = events.find((e) => e.type === 'done');
        assert.ok(done, 'must have a done event');
        assert.ok(done.message.usage.outputTokens <= 10, `generated too many tokens: ${done.message.usage.outputTokens}`);
    });
});
