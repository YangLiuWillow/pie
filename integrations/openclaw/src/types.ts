/**
 * Types for the Pie ↔ OpenClaw integration.
 *
 * These mirror the subset of OpenClaw's llm-core types that the Pie
 * provider needs to consume (Context, Message, Tool) and produce
 * (AssistantMessageEvent, ToolCall, Usage).
 *
 * We define them locally so this package has zero dependency on OpenClaw's
 * internals at build time — the extension host passes them in at runtime
 * through the StreamFunction contract.
 */

// ─── OpenClaw input types (what the extension receives) ────────────────

export interface Context {
    systemPrompt?: string;
    messages: Message[];
    tools?: Tool[];
}

export type Message = UserMessage | AssistantMessage | ToolResultMessage;

export interface UserMessage {
    role: 'user';
    content: string;
    timestamp?: number;
}

export interface AssistantMessage {
    role: 'assistant';
    content: ContentBlock[];
    stopReason?: StopReason;
    usage?: Usage;
    timestamp?: number;
}

export interface ToolResultMessage {
    role: 'tool';
    toolCallId: string;
    content: string;
}

export type ContentBlock = TextContent | ToolCallContent;

export interface TextContent {
    type: 'text';
    text: string;
}

export interface ToolCallContent {
    type: 'tool_call';
    toolCall: ToolCall;
}

export interface ToolCall {
    id: string;
    name: string;
    arguments: Record<string, unknown>;
}

export interface Tool {
    type: 'function';
    function: {
        name: string;
        description?: string;
        parameters?: Record<string, unknown>;
    };
}

// ─── OpenClaw output types (what the extension emits) ──────────────────

export type StopReason = 'stop' | 'length' | 'toolUse' | 'error' | 'aborted';

export interface Usage {
    inputTokens: number;
    outputTokens: number;
}

export type AssistantMessageEvent =
    | { type: 'start'; partial: Partial<AssistantMessage> }
    | { type: 'text_delta'; contentIndex: number; delta: string }
    | { type: 'toolcall_end'; contentIndex: number; toolCall: ToolCall }
    | { type: 'done'; reason: StopReason; message: AssistantMessage }
    | { type: 'error'; reason: 'aborted' | 'error'; error: AssistantMessage };

// ─── Pie inferlet event protocol (JSON lines on stdout) ────────────────

export type PieStreamEvent =
    | { type: 'text_delta'; delta: string }
    | { type: 'tool_call'; id: string; name: string; arguments: string }
    | { type: 'done'; stop_reason: string; prompt_tokens: number; generated_tokens: number; session_id?: string; turn_message_count?: number };

// ─── Provider config ───────────────────────────────────────────────────

export interface PieProviderConfig {
    pieUri: string;
    pieToken?: string;
    inferlet: string;
    inferletVersion?: string;
}
