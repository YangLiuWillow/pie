/**
 * OpenClaw extension setup for the Pie provider.
 *
 * Handles provider registration and configuration. When OpenClaw calls
 * `buildPieProvider()`, we return the StreamFunction that routes inference
 * through a Pie server.
 *
 * Configuration lives in ~/.openclaw/openclaw.json under the "pie" provider:
 *
 *   {
 *     "providers": {
 *       "pie": {
 *         "pieUri": "ws://127.0.0.1:8080",
 *         "pieToken": "...",
 *         "inferlet": "openclaw-chat",
 *         "inferletVersion": "0.1.0"
 *       }
 *     }
 *   }
 */

import { PieClient } from '@pie-project/client';
import { createPieStream } from './stream.ts';
import type { PieProviderConfig } from './types.ts';

const DEFAULT_CONFIG: PieProviderConfig = {
    pieUri: 'ws://127.0.0.1:8080',
    inferlet: 'openclaw-chat',
};

/**
 * Build the Pie provider. Returns the StreamFunction for OpenClaw's
 * agent-core to call on each LLM request.
 */
export function buildPieProvider(userConfig?: Partial<PieProviderConfig>) {
    const config: PieProviderConfig = { ...DEFAULT_CONFIG, ...userConfig };
    return {
        stream: createPieStream(config),
        config,
    };
}

/**
 * Non-interactive configuration: verify the Pie server is reachable and
 * the inferlet is installed.
 */
export async function configurePieNonInteractive(
    config?: Partial<PieProviderConfig>,
): Promise<{ config: PieProviderConfig; ready: boolean; error?: string }> {
    const resolved: PieProviderConfig = { ...DEFAULT_CONFIG, ...config };

    const client = new PieClient(resolved.pieUri);
    try {
        await client.connect();
        if (resolved.pieToken) {
            await client.authByToken(resolved.pieToken);
        }
        await client.ping();

        // Check that the inferlet is installed.
        const inferletFull = resolved.inferletVersion
            ? `${resolved.inferlet}@${resolved.inferletVersion}`
            : resolved.inferlet;

        if (resolved.inferletVersion) {
            const exists = await client.checkProgram(inferletFull);
            if (!exists) {
                return {
                    config: resolved,
                    ready: false,
                    error: `Inferlet '${inferletFull}' is not installed on the Pie server. Run: pie install ${resolved.inferlet}`,
                };
            }
        }

        return { config: resolved, ready: true };
    } catch (e) {
        return {
            config: resolved,
            ready: false,
            error: `Cannot reach Pie server at ${resolved.pieUri}: ${e instanceof Error ? e.message : String(e)}`,
        };
    } finally {
        await client.close();
    }
}
