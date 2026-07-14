/**
 * Unit tests for the OpenClaw ↔ Pie message/tool conversion logic.
 *
 * These tests are hermetic — no Pie server needed.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

// We test the conversion functions indirectly by importing the stream
// module and checking that the exported createPieStream produces a
// valid async generator function. For deeper conversion tests we'd
// export the helpers; for now we validate the public API shape.

// Node 24 strips types natively — import .ts directly.
import { createPieStream, buildPieProvider } from '../src/index.ts';  // eslint-disable-line

describe('createPieStream', () => {
    it('returns an async generator function', () => {
        const stream = createPieStream({
            pieUri: 'ws://127.0.0.1:9999',
            inferlet: 'openclaw-chat',
        });
        assert.equal(typeof stream, 'function');
    });
});

describe('buildPieProvider', () => {
    it('returns a provider with stream and config', () => {
        const provider = buildPieProvider({
            pieUri: 'ws://localhost:8080',
            pieToken: 'test-token',
        });
        assert.ok(provider.stream, 'must have a stream function');
        assert.ok(provider.config, 'must have a config object');
        assert.equal(provider.config.pieUri, 'ws://localhost:8080');
        assert.equal(provider.config.pieToken, 'test-token');
        assert.equal(provider.config.inferlet, 'openclaw-chat');
    });

    it('uses defaults when no config is provided', () => {
        const provider = buildPieProvider();
        assert.equal(provider.config.pieUri, 'ws://127.0.0.1:8080');
        assert.equal(provider.config.inferlet, 'openclaw-chat');
        assert.equal(provider.config.pieToken, undefined);
    });
});
