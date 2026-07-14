/**
 * Pie StreamFunction adapter for OpenClaw.
 *
 * Implements the (model, context, options?) → AssistantMessageEventStream
 * contract that OpenClaw's agent-core expects from an LLM provider.
 *
 * Flow:
 *   1. Convert OpenClaw Context → JSON input for the openclaw-chat inferlet.
 *   2. Launch a Pie process via PieClient.launchProcess().
 *   3. Read process events (stdout lines), parse the JSON-line protocol,
 *      and yield OpenClaw AssistantMessageEvents.
 *
 * Session persistence:
 *   Pass a `sessionId` in StreamOptions to enable KV cache pinning. The
 *   inferlet saves its context after each turn, and on follow-up turns
 *   only the new messages are appended (skipping a full replay).
 */

import { PieClient } from '@pie-project/client';
import type {
    AssistantMessage,
    AssistantMessageEvent,
    ContentBlock,
    Context,
    Message,
    PieProviderConfig,
    PieStreamEvent,
    StopReason,
    TextContent,
    Tool,
    ToolCall,
    ToolCallContent,
    Usage,
} from './types.ts';

interface StreamOptions {
    temperature?: number;
    maxTokens?: number;
    stop?: string[];
    sessionId?: string;
    resumeFrom?: number;
}

/**
 * Per-session state tracked across turns. The stream function updates
 * this after each successful turn so the next call can resume.
 */
export interface SessionState {
    sessionId: string;
    turnMessageCount: number;
}

/**
 * Build a StreamFunction bound to a specific Pie server and inferlet.
 *
 * The returned function also exposes a `lastSession` property that is
 * updated after each turn — callers can read it to pass `resumeFrom`
 * on the next invocation.
 */
export function createPieStream(config: PieProviderConfig) {
    let lastSession: SessionState | null = null;

    const pieStream = async function* (
        _model: unknown,
        context: Context,
        options?: StreamOptions,
    ): AsyncGenerator<AssistantMessageEvent> {
        const client = new PieClient(config.pieUri);
        try {
            await client.connect();
            if (config.pieToken) {
                await client.authByToken(config.pieToken);
            }

            const inferletName = config.inferletVersion
                ? `${config.inferlet}@${config.inferletVersion}`
                : config.inferlet;

            const input = buildInferletInput(context, options);
            const process = await client.launchProcess(inferletName, input);

            const contentBlocks: ContentBlock[] = [];
            let currentTextIndex = -1;
            let accumulatedText = '';
            let usage: Usage = { inputTokens: 0, outputTokens: 0 };
            let finalStopReason: StopReason = 'stop';

            yield {
                type: 'start',
                partial: { role: 'assistant', content: [] },
            };

            while (true) {
                const { event, value } = await process.recv();

                if (event === 'output' && typeof value === 'string') {
                    for (const line of value.split('\n')) {
                        const trimmed = line.trim();
                        if (!trimmed) continue;

                        let pieEvent: PieStreamEvent;
                        try {
                            pieEvent = JSON.parse(trimmed);
                        } catch {
                            continue;
                        }

                        if (pieEvent.type === 'text_delta') {
                            if (currentTextIndex < 0) {
                                currentTextIndex = contentBlocks.length;
                                contentBlocks.push({ type: 'text', text: '' });
                            }
                            accumulatedText += pieEvent.delta;
                            (contentBlocks[currentTextIndex] as TextContent).text = accumulatedText;

                            yield {
                                type: 'text_delta',
                                contentIndex: currentTextIndex,
                                delta: pieEvent.delta,
                            };
                        } else if (pieEvent.type === 'tool_call') {
                            let args: Record<string, unknown>;
                            try {
                                args = JSON.parse(pieEvent.arguments);
                            } catch {
                                args = {};
                            }

                            const toolCall: ToolCall = {
                                id: pieEvent.id,
                                name: pieEvent.name,
                                arguments: args,
                            };

                            const idx = contentBlocks.length;
                            contentBlocks.push({ type: 'tool_call', toolCall });

                            yield {
                                type: 'toolcall_end',
                                contentIndex: idx,
                                toolCall,
                            };
                        } else if (pieEvent.type === 'done') {
                            finalStopReason = mapStopReason(pieEvent.stop_reason);
                            usage = {
                                inputTokens: pieEvent.prompt_tokens,
                                outputTokens: pieEvent.generated_tokens,
                            };
                            if (pieEvent.session_id) {
                                lastSession = {
                                    sessionId: pieEvent.session_id,
                                    turnMessageCount: pieEvent.turn_message_count ?? 0,
                                };
                            }
                        }
                    }
                } else if (event === 'return' || event === 'error') {
                    break;
                }
            }

            const finalMessage: AssistantMessage = {
                role: 'assistant',
                content: contentBlocks,
                stopReason: finalStopReason,
                usage,
                timestamp: Date.now(),
            };

            yield { type: 'done', reason: finalStopReason, message: finalMessage };
        } finally {
            await client.close();
        }
    };

    return Object.assign(pieStream, {
        get lastSession() { return lastSession; },
    });
}

// ─── Conversion helpers ────────────────────────────────────────────────

function buildInferletInput(context: Context, options?: StreamOptions): Record<string, unknown> {
    const messages = convertMessages(context.systemPrompt, context.messages);
    const tools = convertTools(context.tools);

    const input: Record<string, unknown> = {
        messages: JSON.stringify(messages),
        tools: tools.length > 0 ? JSON.stringify(tools) : undefined,
        max_tokens: options?.maxTokens ?? 4096,
        temperature: options?.temperature ?? 0.0,
        top_p: 0.95,
        stop: options?.stop ? JSON.stringify(options.stop) : undefined,
    };

    if (options?.sessionId) {
        input.session_id = options.sessionId;
    }
    if (options?.resumeFrom != null && options.resumeFrom > 0) {
        input.resume_from = options.resumeFrom;
    }

    return input;
}

interface OpenAIMessage {
    role: string;
    content?: string;
    tool_calls?: Array<{ function: { name: string; arguments: string } }>;
    tool_call_id?: string;
}

function convertMessages(systemPrompt: string | undefined, messages: Message[]): OpenAIMessage[] {
    const out: OpenAIMessage[] = [];

    if (systemPrompt) {
        out.push({ role: 'system', content: systemPrompt });
    }

    for (const msg of messages) {
        if (msg.role === 'user') {
            out.push({ role: 'user', content: msg.content });
        } else if (msg.role === 'assistant') {
            const text = msg.content
                .filter((b): b is TextContent => b.type === 'text')
                .map((b) => b.text)
                .join('');

            const toolCalls = msg.content
                .filter((b): b is ToolCallContent => b.type === 'tool_call')
                .map((b) => ({
                    function: {
                        name: b.toolCall.name,
                        arguments: JSON.stringify(b.toolCall.arguments),
                    },
                }));

            const oaiMsg: OpenAIMessage = { role: 'assistant' };
            if (text) oaiMsg.content = text;
            if (toolCalls.length > 0) oaiMsg.tool_calls = toolCalls;
            out.push(oaiMsg);
        } else if (msg.role === 'tool') {
            out.push({
                role: 'tool',
                content: msg.content,
                tool_call_id: msg.toolCallId,
            });
        }
    }

    return out;
}

interface OpenAITool {
    function: {
        name: string;
        description: string;
        parameters: Record<string, unknown>;
    };
}

function convertTools(tools?: Tool[]): OpenAITool[] {
    if (!tools) return [];
    return tools.map((t) => ({
        function: {
            name: t.function.name,
            description: t.function.description ?? '',
            parameters: t.function.parameters ?? {},
        },
    }));
}

function mapStopReason(pieReason: string): StopReason {
    switch (pieReason) {
        case 'stop':
        case 'eos':
            return 'stop';
        case 'length':
            return 'length';
        case 'tool_calls':
            return 'toolUse';
        default:
            return 'stop';
    }
}
