"""PieLLM: an openhands.sdk.LLM that routes the LiteLLM HTTP call through Pie.

We keep all of OpenHands' message formatting, retry decoration, telemetry,
and LLMResponse construction. We only override `_transport_call`, which is
where the parent class would otherwise call `litellm.completion(...)`.
Instead, we launch a Pie inferlet that consumes the structured message/tool
history and returns generated text plus any native tool calls.

The inferlet (not PieLLM) owns chat-template rendering and history replay —
see `inferlets/openhands-completion/src/lib.rs` and
`integrations/openhands/docs/TOOL_CALL_HISTORY_REPLAY_DESIGN.md`.

See: pie/integrations/openhands/docs/SDK_INTERNALS.md  §1.2
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from typing import Any

from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message as LiteLLMMessage,
    ModelResponse,
    Usage,
)
from openhands.sdk.llm import LLM
from openhands.sdk.llm.streaming import TokenCallbackType
from pydantic import Field, PrivateAttr
from pie_client import Event, PieClient

# Phase names in the inferlet's own `Timings` struct
# (inferlets/openhands-coder-session/src/lib.rs:207). `total_ms` is the whole
# inferlet body, entry to exit — it does NOT include anything before entry.
INFERLET_PHASES = ("setup_ms", "render_ms", "hash_ms", "open_ms",
                   "prefill_ms", "save_ms", "fork_ms", "decode_ms", "total_ms")

# Phase names in `_call_pie`'s `_transport` block. These cover exactly the part
# the inferlet cannot see: opening the websocket, authenticating, launching the
# wasm process, waiting for admission, and tearing the connection down.
# `connect`/`auth`/`launch`/`close` appear only in one-shot mode — in daemon
# mode they are paid once at boot, not per call, and their absence from a run's
# telemetry is the headline result of the daemon change. `signal_ms` is the
# daemon-only cost of handing the request to an already-running process.
TRANSPORT_PHASES = ("connect_ms", "auth_ms", "launch_ms", "signal_ms",
                    "first_event_ms", "wait_ms", "close_ms")


def _as_text(value: Any) -> str:
    """Decode a websocket frame that may arrive as bytes or str."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return value if isinstance(value, str) else str(value)


def summarize_pie_call_timings(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-call latency attribution across one or more PieLLMs.

    Returns ``{}`` when nothing was recorded (e.g. a mocked transport in tests
    or a non-Pie arm), so callers can treat a non-empty result as "this arm ran
    on Pie and the attribution is trustworthy".
    """
    if not calls:
        return {}

    def agg(key: str) -> dict[str, float] | None:
        xs = [c[key] for c in calls if key in c]
        if not xs:
            return None
        xs_sorted = sorted(xs)
        return {
            "total_s": round(sum(xs) / 1000.0, 3),
            "median_ms": round(xs_sorted[len(xs_sorted) // 2], 2),
            "p90_ms": round(
                xs_sorted[min(len(xs_sorted) - 1, int(0.9 * len(xs_sorted)))], 2),
            "max_ms": round(max(xs), 2),
        }

    phases: dict[str, Any] = {}
    for key in ("host_ms", *INFERLET_PHASES, *TRANSPORT_PHASES, "unaccounted_ms"):
        a = agg(key)
        if a is not None:
            phases[key.removesuffix("_ms")] = a

    tokens = sum(c.get("tokens_generated", 0) for c in calls)
    decode_s = sum(c.get("decode_ms", 0.0) for c in calls) / 1000.0
    host_s = sum(c["host_ms"] for c in calls) / 1000.0
    modes: dict[str, int] = {}
    for c in calls:
        m = str(c.get("transport_mode") or "oneshot")
        modes[m] = modes.get(m, 0) + 1
    return {
        "num_calls": len(calls),
        "tokens_generated": tokens,
        # Which transport served these calls. A run showing {"daemon": N} paid
        # connect/auth/launch ONCE; {"oneshot": N} paid them N times.
        "transport_modes": modes,
        "daemon_boot_ms": round(
            sum(c.get("daemon_boot_ms", 0.0) for c in calls), 2),
        "phases": phases,
        # The two throughput numbers that matter, and the gap between them is
        # the finding. `decode_tok_s` is what Pie's kernels actually sustain
        # once the request is in flight; `effective_tok_s` is what the agent
        # experiences. vLLM's per-call latency has no equivalent of the gap,
        # because its server is already running when the request arrives.
        "decode_tok_s": round(tokens / decode_s, 2) if decode_s > 0 else 0.0,
        "effective_tok_s": round(tokens / host_s, 2) if host_s > 0 else 0.0,
        "overhead_fraction": (
            round(sum(c.get("unaccounted_ms", 0.0) for c in calls)
                  / (host_s * 1000.0), 4) if host_s > 0 else 0.0
        ),
        # Per-call rows, so the distribution (the p90/max outliers an aggregate
        # hides — Pie's H200 calls ranged 4 s to 109 s) survives into the
        # predictions file instead of being averaged away.
        "per_call": [
            {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()}
            for c in calls
        ],
    }


class PieLLM(LLM):
    """openhands.sdk.LLM subclass that routes the transport layer to Pie.

    All openhands-sdk machinery (retry, telemetry, message formatting, native
    tool calling, prompt caching markers, fallback strategies) continues to
    work — we hook in *below* it.
    """

    # ------- Pie-specific config (extra fields on top of LLM) ----------
    pie_uri: str = Field(
        default="ws://127.0.0.1:8080",
        description="Pie server WebSocket URI",
    )
    pie_username: str = Field(
        default="local-dev",
        description="Pie auth username (see `pie auth`)",
    )
    pie_inferlet: str = Field(
        default="openhands-completion",
        description="Inferlet name registered with `pie serve`",
    )
    pie_request_timeout_s: float = Field(
        default=600.0,
        description="Hard timeout per Pie request",
    )
    pie_session: bool = Field(
        default=False,
        description="Keep the conversation's prompt KV alive across calls via "
                    "a named Pie context (requires a session-capable inferlet, "
                    "e.g. openhands-coder-session). Each call then prefills "
                    "only the token delta since the previous call; on any "
                    "history rewrite the inferlet rebuilds from scratch, so "
                    "semantics are unchanged either way.",
    )
    pie_kv_verify: bool = Field(
        default=False,
        description="Fidelity mode: the inferlet asserts on every call that "
                    "the session context's accumulated tokens equal the "
                    "from-scratch prompt render (Phase 2 of the coder-session "
                    "design). Errors out instead of proceeding on mismatch.",
    )
    pie_use_grammar: bool = Field(
        default=True,
        description="Constrain tool-call generation with the runtime's "
                    "tool-call grammar. Disable for parity with an "
                    "unconstrained vLLM baseline: the model then emits its "
                    "native ChatML tool-call format and the inferlet's "
                    "decoder parses it without masking. Only honored by "
                    "inferlets that read `use_grammar` (openhands-coder-"
                    "session); openhands-completion always constrains.",
    )
    pie_python_tool_parser: bool = Field(
        default=False,
        description="Parse tool calls host-side from the inferlet's raw "
                    "generation using a verbatim port of vLLM's `qwen3_coder` "
                    "tool parser (the exact parser the litellm baseline runs), "
                    "instead of trusting the inferlet's own Rust decoder. "
                    "Implies unconstrained generation (forces use_grammar=off) "
                    "and skips the JSON-format few-shot examples, so the model "
                    "emits its native Qwen3-Coder XML `<function=…>` format and "
                    "we parse it exactly as the baseline does. Maximizes "
                    "tool-call parity. Cache reuse is unaffected: the inferlet "
                    "still snapshots re-rendered prompt tokens, so parsing is "
                    "post-generation and host-side.",
    )

    pie_daemon: bool = Field(
        default=True,
        description="Keep ONE inferlet process and websocket alive for the "
                    "whole conversation and feed it requests, instead of "
                    "launching a fresh process per LLM call. This is what "
                    "gives Pie the same process lifetime vLLM's HTTP server "
                    "has; without it a Pie-vs-vLLM latency comparison is "
                    "measuring this integration's call pattern rather than "
                    "either serving stack. Set False to reproduce the "
                    "pre-2026-07-28 launch-per-call transport.",
    )
    pie_daemon_boot_timeout_s: float = Field(
        default=900.0,
        description="Seconds to wait for the daemon inferlet to report ready. "
                    "Generous because a cold pie server is still loading ~58 GB "
                    "of weights when the first conversation starts.",
    )

    # Session bookkeeping. `session_id` is the only thing that goes on the
    # wire: the inferlet's prefix cache self-keys from the token content it
    # renders, so the host never tells it where the previous turn ended. The
    # length/hash below are what the inferlet *reported* back, kept for
    # telemetry (`pie_session_summary`) and debugging only.
    _pie_session_id: str | None = PrivateAttr(default=None)
    _pie_session_len: int = PrivateAttr(default=0)
    _pie_session_hash: str | None = PrivateAttr(default=None)
    _pie_session_stats: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    # Per-call latency attribution, recorded UNCONDITIONALLY on every Pie call.
    #
    # This used to exist only behind `PIE_DEBUG_LOG` (see `_maybe_debug_dump`),
    # which meant the 2026-07-27 H200 arms — the ones that produced the headline
    # 16x gap against vLLM — recorded no attribution at all. The single most
    # important question about those numbers (how much of Pie's ~11 s median
    # call is GPU work vs the per-call connect/auth/launch/teardown that
    # `_call_pie` pays and a persistent HTTP server does not) was therefore
    # unanswerable from the collected data.
    #
    # It is cheap: six `time.monotonic()` reads already taken by `_call_pie`,
    # plus a dict append per call. There is no reason for it to be opt-in.
    _pie_call_timings: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    # Daemon-transport handles. All None until the first call boots them.
    _pie_loop: Any = PrivateAttr(default=None)
    _pie_thread: Any = PrivateAttr(default=None)
    _pie_client_cm: Any = PrivateAttr(default=None)
    _pie_client: Any = PrivateAttr(default=None)
    _pie_proc: Any = PrivateAttr(default=None)
    _pie_daemon_boot_ms: float = PrivateAttr(default=0.0)

    # `_wrap_as_model_response` now populates real `tool_calls` from the
    # inferlet's structured output (see `assistant_with_tool_calls`/
    # `answer_batch` on the runtime's `Instruct` trait), so the base class's
    # native tool-calling path is used instead of its prompt-mocked one.

    # `LLM.model_config` already sets extra='ignore', so unknown kwargs in
    # base_url / api_key / api_version that we don't use here are harmless.

    # ------------------------------------------------------------------
    # The override
    # ------------------------------------------------------------------
    def _transport_call(
        self,
        *,
        messages: list[dict[str, Any]],
        enable_streaming: bool = False,
        on_token: TokenCallbackType | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Replace litellm.completion(...) with a Pie inferlet round-trip.

        ``messages`` arrives already formatted as OpenAI-style chat dicts
        (the parent class ran format_messages_for_llm before us). We forward
        them — plus any ``tools`` — to the inferlet as structured JSON; the
        inferlet does its own chat-template rendering and history replay.

        Streaming is **not** supported yet — we raise if requested.
        """
        if enable_streaming:
            _ = on_token  # signature parity with parent; consumed in a later phase
            raise NotImplementedError(
                "PieLLM does not support streaming yet. "
                "Wire on_token through the inferlet's session.send chunks."
            )

        wire_messages = [_flatten_content(m) for m in messages]
        # In non-native mode the SDK embeds tool descriptions in the prompt
        # text and passes tools in a flat format the inferlet can't parse.
        tools = (kwargs.get("tools") or []) if self.native_tool_calling else []

        # The Python-side parser reproduces the vLLM baseline's tool-calling
        # path: unconstrained generation + native Qwen3-Coder XML, no JSON
        # few-shot nudges (which would push the model off the XML format the
        # parser expects).
        if tools and self.native_tool_calling and not self.pie_python_tool_parser:
            _inject_native_examples(wire_messages, tools)

        gen_params = self._extract_gen_params(kwargs)
        if not self.pie_use_grammar or self.pie_python_tool_parser:
            gen_params["use_grammar"] = False
        if self.pie_session:
            gen_params.update(self._session_request_fields())
        # Host-side wallclock around the whole Pie round trip. The inferlet's
        # own `timings.total_ms` starts at inferlet entry, so it cannot see
        # process launch, the request/response hop, or serializing the message
        # history. `host_ms - timings.total_ms` is exactly that transport cost,
        # which job 19245724 left as the open question: the inferlet accounted
        # for 4.35 s of a call while the instance wallclock implied far more.
        _t0 = time.monotonic()
        raw = self._invoke_pie(wire_messages, tools, gen_params)
        _host_ms = (time.monotonic() - _t0) * 1000.0
        _maybe_debug_dump(raw, _host_ms)
        self._record_call_timing(raw, _host_ms)
        if self.pie_session:
            self._record_session_response(raw)
        if self.pie_python_tool_parser:
            _reparse_tool_calls_python(raw, tools)
        _sanitize_tool_args(raw, tools)
        return self._wrap_as_model_response(raw)

    # ------------------------------------------------------------------
    # Session protocol (openhands-coder-session inferlet)
    # ------------------------------------------------------------------
    def _session_request_fields(self) -> dict[str, Any]:
        if self._pie_session_id is None:
            self._pie_session_id = uuid.uuid4().hex
        # The coder-session inferlet now self-keys its KV prefix cache from the
        # token content (content-addressed `apc/{session_id}/…` snapshots), so
        # it no longer needs the host to echo the previous render's length/hash
        # or the parent fork pointers — `session_id` alone is the cache
        # namespace. The legacy `session_prev_*` / `session_fork_*` hints are
        # therefore no longer sent (the inferlet would ignore them). Strong
        # cross-agent (delegation) reuse under content addressing is a
        # follow-up: it needs the child to share the parent's namespace AND the
        # shared-prefix boundary to have been saved — see model_copy.
        fields: dict[str, Any] = {"session_id": self._pie_session_id}
        if self.pie_kv_verify:
            fields["kv_verify"] = True
        # Live-context capability experiment (env-gated, off by default): the
        # inferlet holds a live Context across calls and lets the runtime's
        # market scheduler (bid-ordered swap/restore) handle KV pressure,
        # instead of the snapshot save/open cycle. Arms run this way are NOT
        # comparable to snapshot-mode arms — the pie_session.mode strings in
        # the predictions rows carry a "live-" prefix so rows self-identify.
        if os.environ.get("PIE_LIVE_CONTEXT", "") not in ("", "0"):
            fields["live_context"] = True
            if os.environ.get("PIE_LIVE_IDLE_SUSPEND", "") not in ("", "0"):
                fields["live_idle_suspend"] = True
        return fields

    def model_copy(self, *, update: Any = None, deep: bool = False) -> "PieLLM":
        """Give a delegated sub-agent's LLM its own session identity.

        The SDK builds a delegated sub-agent's LLM by ``model_copy``-ing the
        parent's (see ``openhands.tools.task.manager``). Pydantic copies our
        private attrs verbatim, so without intervention the child would inherit
        the parent's ``_pie_session_id``, extend the parent's cache namespace,
        and interleave two conversations in one. Clearing the id hands the
        child a namespace of its own.

        The child still reuses the parent's KV — it just needs no help doing
        so. Its first render shares the task prefix, and identical tokens hash
        to the name the parent already saved, so the content-addressed lookup
        hits it. That is why there is no fork handshake here: an earlier
        version passed the parent's id and render length along, and self-keyed
        caching made every one of those hints redundant.

        Idempotent across the SDK's double copy (parent→child, then a second
        copy that flips ``stream``), since clearing an already-cleared id is a
        no-op.
        """
        child = super().model_copy(update=update, deep=deep)
        # Shallow copy aliases the parent's telemetry lists — give fresh ones.
        child._pie_session_stats = []
        child._pie_call_timings = []
        # The child must NOT inherit the parent's daemon handles. Pydantic
        # copies private attrs verbatim, so without this the condenser would
        # share the agent's inferlet process and websocket: two conversations
        # interleaved on one context, and whichever finished first would tear
        # the transport out from under the other. Clearing them makes the child
        # boot its own daemon on first use, mirroring how it gets its own
        # session id above.
        child._pie_loop = None
        child._pie_thread = None
        child._pie_client_cm = None
        child._pie_client = None
        child._pie_proc = None
        child._pie_daemon_boot_ms = 0.0
        # The child must not share the parent's session identity.
        child._pie_session_id = None
        child._pie_session_len = 0
        child._pie_session_hash = None
        return child

    def _record_session_response(self, raw: dict[str, Any]) -> None:
        session = raw.get("session")
        if not isinstance(session, dict):
            return
        self._pie_session_len = int(session.get("len") or 0)
        self._pie_session_hash = session.get("hash") or None
        self._pie_session_stats.append({
            "mode": session.get("mode"),
            "prompt_len": self._pie_session_len,
            "prefill_tokens": int(session.get("prefill_tokens") or 0),
        })

    def close_pie_session(self) -> None:
        """Delete the server-side session context (idempotent, never raises).

        Call when the conversation ends — a leaked session would pin its KV
        snapshot until an external sweep removes it.
        """
        try:
            if self.pie_session and self._pie_session_id is not None:
                # Route the delete over whatever transport is live. Going
                # straight to `_call_pie` here would launch a fresh one-shot
                # process even in daemon mode — harmless but it would put a
                # spurious launch in the telemetry of every conversation's
                # last moments.
                self._invoke_pie(
                    [], [],
                    {"session_id": self._pie_session_id,
                     "session_action": "delete"},
                )
        except Exception:
            pass
        finally:
            self._pie_session_id = None
            self._pie_session_len = 0
            self._pie_session_hash = None
            # The daemon outlives individual calls by design, so it has to be
            # closed explicitly and AFTER the session delete above — shutting
            # it first would leave the KV snapshots pinned for the lifetime of
            # the pie server.
            self.close_pie_daemon()

    def pie_session_summary(self) -> dict[str, Any]:
        """Aggregate per-call session telemetry for benchmark metadata."""
        stats = self._pie_session_stats
        modes: dict[str, int] = {}
        for s in stats:
            m = str(s.get("mode"))
            modes[m] = modes.get(m, 0) + 1
        return {
            "num_calls": len(stats),
            "prompt_tokens_rendered": sum(s["prompt_len"] for s in stats),
            "prompt_tokens_prefilled": sum(s["prefill_tokens"] for s in stats),
            "modes": modes,
        }

    # ------------------------------------------------------------------
    # Per-call latency attribution
    # ------------------------------------------------------------------
    # The inferlet's own clock (`timings`) starts at its entry point, so it
    # cannot see connection setup, authentication, process launch, admission
    # queueing, or teardown. `host_ms` brackets the whole round trip, and the
    # difference between them is what THIS INTEGRATION pays per call while
    # vLLM's persistent HTTP server does not.
    #
    # Be precise about whose cost that is. It is not a property of Pie: the
    # platform supports a long-lived inferlet that is launched once and fed
    # many requests (`session::receive` / `session::send`,
    # runtime/wit/core/wit/session.wit:9, driven from the client by
    # `Process.signal`, client/python/src/pie_client/client.py:40 — see
    # inferlets/text-completion-bench/src/lib.rs:130 for an inferlet that does
    # exactly this). It is a property of `_call_pie` below, which opens a
    # websocket, authenticates, launches a fresh wasm process and tears it all
    # down inside every single `_transport_call`.
    #
    # So this measurement attributes OUR design, not Pie's kernels, and must be
    # subtracted before any "vLLM is Nx faster than Pie" claim is made.
    def _record_call_timing(self, raw: Any, host_ms: float) -> None:
        """Record one call's timing breakdown. Never raises — telemetry must
        not be able to fail a benchmark run."""
        try:
            timings = raw.get("timings") if isinstance(raw, dict) else None
            transport = raw.get("_transport") if isinstance(raw, dict) else None
            timings = timings if isinstance(timings, dict) else {}
            transport = transport if isinstance(transport, dict) else {}
            rec: dict[str, Any] = {"host_ms": float(host_ms)}
            for k in INFERLET_PHASES:
                if k in timings:
                    rec[k] = float(timings[k] or 0.0)
            for k in TRANSPORT_PHASES:
                if k in transport:
                    rec[k] = float(transport[k] or 0.0)
            if "open_attempts" in timings:
                rec["open_attempts"] = int(timings["open_attempts"] or 0)
            if isinstance(raw, dict):
                rec["tokens_generated"] = int(raw.get("tokens_generated") or 0)
                rec["transport_mode"] = raw.get("_transport_mode") or "oneshot"
            # One-time daemon boot, attributed to the call that paid for it
            # rather than folded into its latency.
            if self._pie_daemon_boot_ms and not self._pie_call_timings:
                rec["daemon_boot_ms"] = self._pie_daemon_boot_ms
            # host_ms minus the inferlet's own total. Positive by construction;
            # a negative value would mean the two clocks disagree and is worth
            # seeing rather than clamping.
            if "total_ms" in rec:
                rec["unaccounted_ms"] = rec["host_ms"] - rec["total_ms"]
            self._pie_call_timings.append(rec)
        except Exception:  # pragma: no cover — defensive
            pass

    def pie_timing_summary(self) -> dict[str, Any]:
        """This LLM's own per-call latency attribution.

        Use :func:`summarize_pie_call_timings` directly to merge several LLMs
        (agent + condenser), which is what the benchmark harness does.
        """
        return summarize_pie_call_timings(self._pie_call_timings)

    # ------------------------------------------------------------------
    # Parameter extraction from kwargs
    # ------------------------------------------------------------------
    def _extract_gen_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "max_tokens": int(kwargs.get("max_tokens") or self.max_output_tokens or 2048),
            "temperature": float(kwargs.get("temperature")
                                 if kwargs.get("temperature") is not None
                                 else (self.temperature if self.temperature is not None else 0.0)),
            "top_p": float(kwargs.get("top_p") or 0.95),
            "stop": list(kwargs.get("stop") or []),
            "model": self.model,
        }

    # ------------------------------------------------------------------
    # Transport selection: persistent daemon (default) or launch-per-call
    # ------------------------------------------------------------------
    def _invoke_pie(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        gen_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Run one request over whichever transport is configured."""
        if not self.pie_daemon:
            return asyncio.run(self._call_pie(messages, tools, gen_params))
        payload = {"messages": messages, "tools": tools, **gen_params}
        return self._daemon_call(payload)

    # ------------------------------------------------------------------
    # Persistent-daemon transport
    # ------------------------------------------------------------------
    # WHY A BACKGROUND THREAD. `_transport_call` is synchronous — that is the
    # SDK's contract, not our choice — so the one-shot path reaches async code
    # through `asyncio.run(...)`, which builds an event loop, runs one
    # coroutine, and destroys the loop. A websocket cannot outlive that, which
    # is the mechanical reason the old transport had to reconnect, re-auth and
    # relaunch the inferlet on every single call. Owning one long-lived loop on
    # a daemon thread, and submitting work to it with
    # `run_coroutine_threadsafe`, is what lets the connection AND the wasm
    # process survive between calls — the same lifetime vLLM's server has.
    def _ensure_daemon(self) -> None:
        """Start the loop thread and the inferlet process. Idempotent."""
        if self._pie_proc is not None:
            return
        if self._pie_loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                name=f"pie-daemon-{id(self):x}",
                daemon=True,
            )
            thread.start()
            self._pie_loop = loop
            self._pie_thread = thread
        t0 = time.monotonic()
        self._run_on_loop(
            self._daemon_connect(), timeout=self.pie_daemon_boot_timeout_s
        )
        # One-time cost, recorded once rather than amortized silently into the
        # first call — otherwise call 1 looks pathological and the mean lies.
        self._pie_daemon_boot_ms = (time.monotonic() - t0) * 1000.0

    def _run_on_loop(self, coro: Any, timeout: float) -> Any:
        if self._pie_loop is None:  # pragma: no cover — guarded by callers
            raise RuntimeError("pie daemon loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._pie_loop)
        try:
            return future.result(timeout=timeout)
        except Exception:
            future.cancel()
            raise

    async def _daemon_connect(self) -> None:
        cm = PieClient(self.pie_uri)
        client = await cm.__aenter__()
        try:
            await client.authenticate(self.pie_username)
            proc = await client.launch_process(
                self.pie_inferlet, input={"daemon": True}
            )
            # The inferlet announces itself once it is serving. Waiting for it
            # here means the first real request does not silently absorb
            # process startup and model binding.
            while True:
                event, value = await asyncio.wait_for(
                    proc.recv(), timeout=self.pie_daemon_boot_timeout_s
                )
                if event == Event.Message:
                    if json.loads(_as_text(value)).get("ready"):
                        break
                elif event == Event.Error:
                    raise RuntimeError(f"pie daemon failed to start: {value!r}")
                elif event == Event.Return:
                    raise RuntimeError(
                        f"pie daemon exited before serving: {value!r}. Is the "
                        "installed inferlet new enough for daemon mode?"
                    )
        except BaseException:
            await cm.__aexit__(None, None, None)
            raise
        self._pie_client_cm = cm
        self._pie_client = client
        self._pie_proc = proc

    def _daemon_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_daemon()
        return self._run_on_loop(
            self._daemon_request(payload),
            timeout=self.pie_request_timeout_s + 60,
        )

    async def _daemon_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        t: dict[str, float] = {}
        clock = time.monotonic
        proc = self._pie_proc
        if proc is None:  # pragma: no cover — _ensure_daemon guarantees this
            raise RuntimeError("pie daemon process is not running")

        t0 = clock()
        await proc.signal(json.dumps(payload))
        t["signal_ms"] = (clock() - t0) * 1000.0

        t0 = clock()
        stdout_chunks: list[str] = []
        first_seen = False
        while True:
            event, value = await asyncio.wait_for(
                proc.recv(), timeout=self.pie_request_timeout_s
            )
            if not first_seen:
                t["first_event_ms"] = (clock() - t0) * 1000.0
                first_seen = True
            if event == Event.Message:
                t["wait_ms"] = (clock() - t0) * 1000.0
                frame = json.loads(_as_text(value))
                if not frame.get("ok"):
                    raise RuntimeError(f"Pie inferlet error: {frame.get('error')!r}")
                out = frame.get("result")
                if not isinstance(out, dict):
                    raise RuntimeError(f"Pie daemon returned a non-object: {out!r}")
                if stdout_chunks:
                    out.setdefault("_stdout", "".join(stdout_chunks))
                # Same key the one-shot path uses, so the timing telemetry and
                # every downstream consumer stay transport-agnostic. `connect`,
                # `auth` and `launch` are absent here by construction — that
                # absence IS the change being measured.
                out["_transport"] = t
                out["_transport_mode"] = "daemon"
                return out
            if event == Event.Stdout:
                stdout_chunks.append(_as_text(value))
            elif event == Event.Error:
                raise RuntimeError(f"Pie inferlet error: {value!r}")
            elif event == Event.Return:
                raise RuntimeError(f"pie daemon exited mid-request: {value!r}")

    def close_pie_daemon(self) -> None:
        """Shut the daemon down. Idempotent, never raises.

        A leaked daemon holds its KV pages and a wasm process for the lifetime
        of the pie server, so this must run on the error path too.
        """
        if self._pie_proc is None and self._pie_loop is None:
            return
        try:
            if self._pie_proc is not None:
                self._run_on_loop(self._daemon_shutdown(), timeout=60)
        except Exception:
            pass
        finally:
            self._pie_proc = None
            self._pie_client = None
            self._pie_client_cm = None
            loop = self._pie_loop
            self._pie_loop = None
            self._pie_thread = None
            if loop is not None:
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except Exception:
                    pass

    async def _daemon_shutdown(self) -> None:
        proc, cm = self._pie_proc, self._pie_client_cm
        try:
            if proc is not None:
                await proc.signal(json.dumps({"daemon_action": "shutdown"}))
                # Give the loop a moment to exit cleanly, then terminate.
                # Either way we stop waiting — shutdown must not hang a
                # benchmark that has already produced its result.
                try:
                    await asyncio.wait_for(proc.recv(), timeout=10)
                except Exception:
                    pass
                try:
                    await proc.terminate()
                except Exception:
                    pass
        finally:
            if cm is not None:
                await cm.__aexit__(None, None, None)

    # ------------------------------------------------------------------
    # The Pie round-trip (one-shot: launch a fresh process per call)
    # ------------------------------------------------------------------
    async def _call_pie(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        gen_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Launch the inferlet, collect its Return payload, return as dict.

        Override this in tests with a mock to avoid spinning up a Pie server.

        Every call opens its own connection, authenticates, and launches a
        fresh process — the inferlet is torn down between calls, which is why
        the KV has to be recovered from a named APC snapshot each turn. Job
        19247969 measured this whole round trip at 13.6 s/call against only
        3.6 s spent inside the inferlet, i.e. ~10 s/call unaccounted for. The
        `_transport` block below splits that 10 s across the individual steps
        so it can be attributed rather than guessed at.
        """
        input_payload = {"messages": messages, "tools": tools, **gen_params}
        t: dict[str, float] = {}
        clock = time.monotonic

        t0 = clock()
        client_cm = PieClient(self.pie_uri)
        client = await client_cm.__aenter__()
        t["connect_ms"] = (clock() - t0) * 1000.0
        try:
            t0 = clock()
            await client.authenticate(self.pie_username)
            t["auth_ms"] = (clock() - t0) * 1000.0

            t0 = clock()
            proc = await client.launch_process(self.pie_inferlet, input=input_payload)
            t["launch_ms"] = (clock() - t0) * 1000.0

            # Time until the *first* event of any kind comes back. The
            # inferlet's own clock starts at its entry point, so anything
            # before that first event is queueing/admission plus request
            # delivery — the part neither side currently sees.
            t0 = clock()
            first_seen = False
            stdout_chunks: list[str] = []
            while True:
                event, value = await asyncio.wait_for(
                    proc.recv(), timeout=self.pie_request_timeout_s
                )
                if not first_seen:
                    t["first_event_ms"] = (clock() - t0) * 1000.0
                    first_seen = True
                if event == Event.Stdout:
                    if isinstance(value, (bytes, bytearray)):
                        stdout_chunks.append(value.decode("utf-8", "replace"))
                    else:
                        stdout_chunks.append(str(value))
                elif event == Event.Return:
                    t["wait_ms"] = (clock() - t0) * 1000.0
                    out = _parse_return_value(value, stdout_chunks)
                    break
                elif event == Event.Error:
                    raise RuntimeError(f"Pie inferlet error: {value!r}")
                # Stderr / Message / File: ignore in Phase 1
        finally:
            t0 = clock()
            await client_cm.__aexit__(None, None, None)
            t["close_ms"] = (clock() - t0) * 1000.0

        if isinstance(out, dict):
            out["_transport"] = t
        return out

    # ------------------------------------------------------------------
    # Pie response -> ModelResponse
    # ------------------------------------------------------------------
    def _wrap_as_model_response(self, pie_out: dict[str, Any]) -> ModelResponse:
        text = pie_out.get("text", "")
        stop_reason = pie_out.get("stop_reason") or "stop"
        prompt_tokens = int(pie_out.get("prompt_tokens") or 0)
        completion_tokens = int(
            pie_out.get("tokens_generated")
            or pie_out.get("completion_tokens")
            or _approx_token_count(text)
        )

        finish_reason = {
            "stop": "stop",
            "eos": "stop",
            "length": "length",
            "tool_calls": "tool_calls",
        }.get(stop_reason, "stop")

        # The inferlet numbers tool calls `call_0`, `call_1`, ... starting
        # over from zero on *every* request (it rebuilds the conversation
        # from scratch each turn — see its own module doc). Since most turns
        # emit exactly one tool call, that means nearly every tool call
        # across an entire multi-turn conversation would otherwise be handed
        # the identical id "call_0". OpenHands SDK's `ObservationUniquenessProperty`
        # dedups observations by `tool_call_id`, so it would then treat every
        # tool result after the first as a duplicate of the same call and
        # silently drop it from the context the model sees — starving the
        # agent of feedback and causing it to repeat the same action forever.
        # A uuid keeps ids unique across the whole conversation, not just
        # within one request.
        tool_calls = [
            ChatCompletionMessageToolCall(
                id=f"call_{uuid.uuid4().hex[:24]}",
                type="function",
                function=Function(name=tc["name"], arguments=tc["arguments"]),
            )
            for tc in (pie_out.get("tool_calls") or [])
        ] or None

        msg = LiteLLMMessage(role="assistant", content=text, tool_calls=tool_calls)
        choice = Choices(index=0, message=msg, finish_reason=finish_reason)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return ModelResponse(
            id=f"pie-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=self.model,
            object="chat.completion",
            choices=[choice],
            usage=usage,
        )


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _flatten_content(message: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *message* with content guaranteed to be a string.

    OpenHands' format_messages_for_llm emits ``content`` as either a string
    or a list of content parts (when vision is active). HF chat templates
    only support string content; collapse parts to their text fields.
    """
    content = message.get("content")
    if isinstance(content, list):
        text = "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
        return {**message, "content": text}
    return message


def _reparse_tool_calls_python(
    pie_out: dict[str, Any],
    tools: list[dict[str, Any]],
) -> None:
    """Re-parse tool calls host-side from the raw generation, in place.

    Replaces the inferlet's Rust-decoded ``tool_calls`` with the result of
    vLLM's ``qwen3_coder`` parser (see ``qwen3coder_parser``) run on the raw
    phase-1 generation. This reproduces the litellm+vLLM baseline's tool-call
    extraction exactly — most importantly its lenient back-off, which recovers
    a ``<function=…>`` call even when the ``<tool_call>`` wrapper is malformed
    or missing (the failure mode that stalled coder-session on django-13028).

    Cache-safety note: we do NOT strip tokenizer special-token strings here.
    The coder-session inferlet's ``sanitize_messages`` already scrubs replayed
    tool-call names/args with the real tokenizer at render time (commit
    d094ed69), which is where the round-trip contract that protects prefix
    reuse actually lives.
    """
    from . import qwen3coder_parser

    raw_text = pie_out.get("debug_full_text")
    if not isinstance(raw_text, str) or not raw_text:
        raw_text = pie_out.get("text") or ""

    parsed = qwen3coder_parser.extract_tool_calls(raw_text, tools)
    if not parsed:
        # No call recovered — leave the inferlet's own tool_calls untouched so
        # we never regress a turn the Rust decoder handled but the port didn't.
        return

    pie_out["tool_calls"] = parsed
    pie_out["stop_reason"] = "tool_calls"

    # Match vLLM's content handling: assistant content is whatever precedes the
    # first tool-call marker (so the XML markup isn't duplicated into content).
    idx_call = raw_text.find("<tool_call>")
    idx_fn = raw_text.find("<function=")
    cut = min(i for i in (idx_call, idx_fn) if i >= 0) if (
        idx_call >= 0 or idx_fn >= 0
    ) else -1
    if cut >= 0:
        pie_out["text"] = raw_text[:cut]


def _sanitize_tool_args(
    pie_out: dict[str, Any],
    tools: list[dict[str, Any]],
) -> None:
    """Strip unknown keys from tool-call arguments in-place.

    Small models sometimes append garbage key-value pairs (e.g. ``", ": ","``
    after the real argument) that survive the layer-1 JSON grammar but get
    rejected by Pydantic ``extra='forbid'`` on OpenHands action models,
    sending the agent into a stuck error loop.
    """
    schema_map: dict[str, set[str]] = {}
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function", {})
        name = fn.get("name", "")
        props = fn.get("parameters", {}).get("properties", {})
        if name and props:
            schema_map[name] = set(props.keys())

    for tc in pie_out.get("tool_calls") or []:
        valid_keys = schema_map.get(tc.get("name", ""))
        if valid_keys is None:
            continue
        try:
            args = json.loads(tc["arguments"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        sanitized = {k: v for k, v in args.items() if k in valid_keys}
        if sanitized != args:
            tc["arguments"] = json.dumps(sanitized)


def _maybe_debug_dump(raw: dict[str, Any], host_ms: float | None = None) -> None:
    """Append the raw inferlet output to ``$PIE_DEBUG_LOG`` (JSONL) if set.

    Temporary diagnostic for the phase-1/phase-2 tool-call investigation:
    surfaces the inferlet's ``debug_full_text`` / ``debug_phase1_marker`` /
    ``debug_phase2_fired`` fields, which the OpenHands completion logger
    drops (it only records the wrapped LiteLLM response).
    """
    path = os.environ.get("PIE_DEBUG_LOG")
    if not path or not isinstance(raw, dict):
        return
    rec = {
        "text": raw.get("text"),
        "tool_calls": raw.get("tool_calls"),
        "stop_reason": raw.get("stop_reason"),
        "debug_phase1_marker": raw.get("debug_phase1_marker"),
        "debug_phase2_fired": raw.get("debug_phase2_fired"),
        "debug_full_text": raw.get("debug_full_text"),
        "session": raw.get("session"),
        # Per-phase wallclock inside the inferlet (setup/render/hash/open/
        # prefill/save/fork/decode/total, ms). Attributes the non-decode time
        # per call; see the Timings struct in the coder-session inferlet.
        "timings": raw.get("timings"),
        # Whole round trip as the host sees it; minus timings.total_ms this is
        # the transport + process-launch cost the inferlet cannot measure.
        "host_ms": host_ms,
        # Actual token counts, so generated-length differences between engines
        # can be normalized instead of inferred from character counts.
        "tokens_generated": raw.get("tokens_generated"),
        "prompt_tokens": raw.get("prompt_tokens"),
        # connect / auth / launch / first_event / wait / close, ms — splits the
        # ~10 s/call that sits between host_ms and the inferlet's total_ms.
        "transport": raw.get("_transport"),
    }
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _parse_return_value(value: Any, stdout_chunks: list[str]) -> dict[str, Any]:
    """Normalize a Pie inferlet's Return payload into a dict.

    Inferlets may return:
      * a JSON-encoded string  -> parse it
      * a dict (already decoded by the client)
      * a plain string         -> treat as ``{"text": value}``
      * None                   -> fall back to concatenated stdout chunks
    """
    if value is None:
        return {"text": "".join(stdout_chunks)}
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {"text": value}
    return {"text": str(value)}


def _approx_token_count(text: str) -> int:
    """Rough character-count-based token estimate for usage fields.

    Real token counts should come from the inferlet's ``Output``. This is
    only a fallback when the inferlet doesn't report them.
    """
    return max(1, len(text) // 4)


# ----------------------------------------------------------------------
# Native-format few-shot examples
# ----------------------------------------------------------------------
# Mirrors OpenHands's non-native ``fn_call_examples`` but uses the native
# ``<tool_call>`` JSON format that Qwen's chat template expects.  Injected
# into the first user message so the model sees concrete examples of
# correct parameter usage — especially ``old_str``/``new_str`` for
# ``file_editor``'s ``str_replace`` command.

_TOOL_CALL = '<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>'

_NATIVE_EXAMPLES: dict[str, dict[str, str]] = {
    "terminal": {
        "check_dir": (
            'ASSISTANT:\n'
            + _TOOL_CALL.format(
                name="terminal",
                args='{"command": "pwd && ls", "security_risk": "LOW", '
                     '"summary": "Check current directory and list files"}',
            )
            + '\n\nUSER: EXECUTION RESULT of [terminal]:\n'
            '/workspace\nopenhands@runtime:~/workspace$\n'
        ),
    },
    "file_editor": {
        "create_file": (
            'ASSISTANT: Let me create the file:\n'
            + _TOOL_CALL.format(
                name="file_editor",
                args='{"command": "create", "path": "/workspace/app.py", '
                     '"file_text": "from flask import Flask\\napp = Flask(__name__)\\n\\n'
                     '@app.route(\\"/\\")\\ndef index():\\n    numbers = list(range(1, 11))\\n'
                     '    return str(numbers)\\n\\nif __name__ == \\"__main__\\":\\n'
                     '    app.run(port=5000)\\n", '
                     '"security_risk": "MEDIUM", '
                     '"summary": "Create Flask app.py with number list endpoint"}',
            )
            + '\n\nUSER: EXECUTION RESULT of [file_editor]:\n'
            'File created successfully at: /workspace/app.py\n'
        ),
        "edit_file": (
            'ASSISTANT: Now let me edit the file:\n'
            + _TOOL_CALL.format(
                name="file_editor",
                args='{"command": "str_replace", "path": "/workspace/app.py", '
                     '"old_str": "return str(numbers)", '
                     '"new_str": "return \'<table>\' + \'\'.join([f\'<tr><td>{i}</td></tr>\' '
                     'for i in numbers]) + \'</table>\'", '
                     '"security_risk": "MEDIUM", '
                     '"summary": "Update return statement to render HTML table"}',
            )
            + '\n\nUSER: EXECUTION RESULT of [file_editor]:\n'
            'The file /workspace/app.py has been edited. Here\'s the result of running `cat -n`:\n'
            '     5\tdef index():\n'
            '     6\t    numbers = list(range(1, 11))\n'
            '     7\t    return \'<table>\' + \'\'.join([f\'<tr><td>{i}</td></tr>\' '
            'for i in numbers]) + \'</table>\'\n'
        ),
    },
    "finish": {
        "example": (
            'ASSISTANT: The task is complete.\n'
            + _TOOL_CALL.format(
                name="finish",
                args='{"message": "I have created the Flask app and updated it to display '
                     'numbers in a table format.", '
                     '"security_risk": "LOW", '
                     '"summary": "Complete task"}',
            )
        ),
    },
}


def _inject_native_examples(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> None:
    """Prepend native-format few-shot examples to the first user message."""
    tool_names = {
        t.get("function", {}).get("name", "") for t in tools if t.get("type") == "function"
    }

    parts: list[str] = []
    if "terminal" in tool_names:
        parts.append(_NATIVE_EXAMPLES["terminal"]["check_dir"])
    if "file_editor" in tool_names:
        parts.append(_NATIVE_EXAMPLES["file_editor"]["create_file"])
    if "file_editor" in tool_names:
        parts.append(_NATIVE_EXAMPLES["file_editor"]["edit_file"])
    if "finish" in tool_names:
        parts.append(_NATIVE_EXAMPLES["finish"]["example"])

    if not parts:
        return

    example_block = (
        "Here's a running example of how to perform a task with the provided tools.\n\n"
        "--------------------- START OF EXAMPLE ---------------------\n\n"
        "USER: Create a list of numbers from 1 to 10, and display them in a web page at port 5000.\n\n"
        + "\n".join(parts)
        + "\n\n--------------------- END OF EXAMPLE ---------------------\n\n"
        "Do NOT assume the environment is the same as in the example above.\n\n"
        "--------------------- NEW TASK DESCRIPTION ---------------------\n"
    )

    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = example_block + content
            elif isinstance(content, list):
                if content and isinstance(content[0], dict) and content[0].get("type") == "text":
                    content[0]["text"] = example_block + content[0].get("text", "")
            break
