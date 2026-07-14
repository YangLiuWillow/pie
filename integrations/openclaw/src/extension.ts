/**
 * OpenClaw extension entry point for the Pie provider.
 *
 * Registers Pie as an LLM provider via OpenClaw's plugin API.
 * OpenClaw discovers this file through the "openclaw.extensions"
 * field in package.json.
 */

import { buildPieProvider, configurePieNonInteractive } from './setup.ts';
import { createPieStream } from './stream.ts';
import type { PieProviderConfig } from './types.ts';

// OpenClaw's plugin SDK types — these are provided at runtime by the host.
// We declare minimal shapes here to avoid a build-time dependency on
// @openclaw/plugin-sdk.
interface OpenClawPluginApi {
    registerProvider(config: ProviderRegistration): void;
}

interface ProviderRegistration {
    id: string;
    name: string;
    description: string;
    createStream(options: { baseUrl?: string; apiKey?: string; model?: string }): ReturnType<typeof createPieStream>;
    authenticate?: {
        nonInteractive(config: Record<string, unknown>): Promise<{ ready: boolean; error?: string }>;
    };
    catalogModels?(): Promise<Array<{ id: string; name: string }>>;
}

interface PluginEntry {
    id: string;
    name: string;
    register(api: OpenClawPluginApi): void;
}

function definePluginEntry(entry: PluginEntry): PluginEntry {
    return entry;
}

// ─── Extension definition ──────────────────────────────────────────────

export default definePluginEntry({
    id: 'pie',
    name: 'Pie Provider',

    register(api: OpenClawPluginApi) {
        api.registerProvider({
            id: 'pie',
            name: 'Pie',
            description: 'Programmable local inference via Pie — custom samplers, tool-call grammars, KV cache control',

            createStream(options) {
                const config: PieProviderConfig = {
                    pieUri: options.baseUrl || 'ws://127.0.0.1:8080',
                    pieToken: options.apiKey,
                    inferlet: 'openclaw-chat',
                };
                return createPieStream(config);
            },

            authenticate: {
                async nonInteractive(config: Record<string, unknown>) {
                    const result = await configurePieNonInteractive({
                        pieUri: (config.baseUrl as string) || undefined,
                        pieToken: (config.apiKey as string) || undefined,
                        inferlet: (config.inferlet as string) || undefined,
                    });
                    return { ready: result.ready, error: result.error };
                },
            },

            async catalogModels() {
                // Pie serves whichever model is loaded — the inferlet name
                // acts as the "model" identifier in OpenClaw's config.
                return [
                    { id: 'pie/openclaw-chat', name: 'Pie (openclaw-chat inferlet)' },
                ];
            },
        });
    },
});
