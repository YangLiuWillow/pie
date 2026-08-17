//! RL rollout inferlet — pre-tokenized prompt in, sampled token ids + exact
//! per-token logprobs out.
//!
//! The generation core of the old `rl-completions` HTTP daemon, re-expressed
//! for the 0.5 architecture (the OpenAI-compatible wire envelope now lives in
//! a host-side bridge; this inferlet is launched once per request). Structure
//! follows `text-completion-bench`: an N-wide chunked prefill fire, then ONE
//! decode pass whose sampled token is device loop-carried, kept ahead of the
//! host drain by the engine-sized run-ahead window.
//!
//! What it adds over the bench: each sampling epilogue also computes
//! `scalar_gather(log_softmax(logits), token)` — the exact logprob of the
//! sampled token under the RAW (unscaled) model distribution, the same
//! quantity a trainer's teacher-forcing recompute produces — and mirrors it
//! to the host on an f32 channel alongside the token. RL contract details:
//! the stop token, when hit, IS included in `token_ids`/`logprobs` (matching
//! vLLM completion accounting, where `usage.completion_tokens` counts it);
//! `text` excludes it.

use inferlet::ptir::attention::prelude::*;
use inferlet::{chat, session};
use serde::{Deserialize, Serialize};
use std::ops::RangeBounds;

#[derive(Deserialize)]
struct Input {
    /// Debug path: plain-text prompt, chat-templated system+user+cue.
    #[serde(default)]
    prompt: String,
    /// The RL path: exact token ids, appended verbatim — no template, no
    /// tokenizer in the loop. Accepts a JSON array or a string containing one
    /// (`pie run` delivers every flag value as a scalar, so the string form is
    /// what CLI invocations produce).
    #[serde(default, deserialize_with = "de_tokens")]
    prompt_tokens: Option<Vec<u32>>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    seed: u32,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default)]
    return_text: bool,
    /// Candidate saved-prefix boundary lengths, longest preferred. Each
    /// candidate is verified content-addressed (hash of `prompt_tokens[..len]`
    /// must hit the index), so a stale hint is a clean miss, never corruption.
    /// The bridge tracks these per conversation lineage; a pressure-evicted
    /// snapshot also just misses. Accepts a JSON array or a string form.
    #[serde(default, deserialize_with = "de_u32_list")]
    saved_lens: Vec<u32>,
    /// Save a snapshot at the completion boundary for the next turn to resume
    /// (attention-only models; hybrids need recurrent-state snapshotting the
    /// index API does not cover).
    #[serde(default = "default_true")]
    save_kv: bool,
    /// Chat path (turn 0): OpenAI-style messages, rendered token-id-native
    /// via chat.wit. When present, wins over prompt/prompt_tokens and the
    /// rendered ids are reported back as `prompt_token_ids`.
    #[serde(default)]
    messages: Option<Vec<ChatMsg>>,
    /// Prefill chunk width, clamped to the driver's `max_embed_length()`.
    /// `None` (the default, and what the bridge always sends) takes the
    /// driver's own capacity, so this changes nothing in production.
    ///
    /// It exists for `test_chunked_prefill.py`: forcing the width down runs the
    /// multi-chunk path on a short prompt, which is the only practical way to
    /// check that concatenating chunks reproduces the one-shot fire. The same
    /// knob is on quest-attention, trackb-h2o, trackb-snapkv and tova; it is
    /// here because this is the inferlet whose LOGPROBS feed RL, and the
    /// existing test asserts only that the generated text matches.
    #[serde(default)]
    prefill_chunk: Option<u32>,
}

#[derive(Deserialize)]
struct ChatMsg {
    role: String,
    /// String, or OpenAI parts array (text parts concatenated).
    #[serde(default)]
    content: serde_json::Value,
}

impl ChatMsg {
    fn text(&self) -> String {
        match &self.content {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        }
    }
}

/// Render an OpenAI message list to token ids via the model's own template
/// (chat.wit — every fill returns ids, so this is the templating the engine
/// itself believes, not a re-implementation).
fn render_messages(messages: &[ChatMsg]) -> ::std::result::Result<Vec<u32>, String> {
    let mut ids: Vec<u32> = Vec::new();
    let mut i = 0;
    if messages.is_empty() {
        return Err("messages must be non-empty".into());
    }
    if messages[0].role == "system" {
        if messages.len() >= 2 && messages[1].role == "user" {
            ids.extend(chat::system_user(&messages[0].text(), &messages[1].text()));
            i = 2;
        } else {
            ids.extend(chat::system(&messages[0].text()));
            i = 1;
        }
    } else if messages[0].role == "user" {
        ids.extend(chat::first_user(&messages[0].text()));
        i = 1;
    }
    for m in &messages[i..] {
        match m.role.as_str() {
            "user" => ids.extend(chat::user(&m.text())),
            "assistant" => ids.extend(chat::assistant(&m.text())),
            other => return Err(format!("unsupported role in position >0: {other}")),
        }
    }
    ids.extend(chat::cue());
    Ok(ids)
}

fn default_true() -> bool {
    true
}

fn default_max_tokens() -> usize {
    256
}
fn default_top_p() -> f32 {
    1.0
}

fn de_tokens<'de, D>(d: D) -> ::std::result::Result<Option<Vec<u32>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
        List(Vec<u32>),
        Text(String),
    }
    match Option::<Raw>::deserialize(d)? {
        None => Ok(None),
        Some(Raw::List(v)) => Ok(Some(v)),
        Some(Raw::Text(s)) => serde_json::from_str(&s)
            .map(Some)
            .map_err(serde::de::Error::custom),
    }
}

fn de_u32_list<'de, D>(d: D) -> ::std::result::Result<Vec<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(de_tokens(d)?.unwrap_or_default())
}

/// FNV-1a-64 over the token ids' little-endian bytes — the same
/// content-address family the 0.4 rl-completions/codex session caches used.
fn fnv1a64(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Index key for a token prefix: tag + length + hash. Length in the key makes
/// collisions across different boundary lengths impossible by construction.
fn index_key(tokens: &[u32]) -> Vec<u8> {
    let mut k = Vec::with_capacity(21);
    k.extend_from_slice(b"rlro1");
    k.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    k.extend_from_slice(&fnv1a64(tokens).to_le_bytes());
    k
}

#[derive(Serialize)]
struct Output {
    num_prompt_tokens: usize,
    /// == token_ids.len(); includes the stop token when finish_reason=="stop".
    num_output_tokens: usize,
    token_ids: Vec<u32>,
    /// log softmax(raw logits)[token] per generated token, index-aligned with
    /// `token_ids`.
    logprobs: Vec<f32>,
    /// "stop" | "length"
    finish_reason: String,
    /// Prompt tokens whose KV came from a resumed snapshot (0 = cold prefill).
    cached_tokens: usize,
    /// Boundary length this run saved for the next turn (0 = nothing saved).
    saved_len: usize,
    /// The rendered prompt ids (messages path only — the bridge already has
    /// them on the token-id path).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    prompt_token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "String::is_empty")]
    text: String,
}

/// In-graph sample + exact logprob from one logits read: greedy argmax at
/// `temperature <= 0`, otherwise temperature-scaled (optionally top-p-masked)
/// Gumbel-max. The logprob is taken from `log_softmax` of the RAW logits —
/// deliberately independent of temperature/top-p, so it matches an external
/// teacher-forcing recompute of the policy distribution.
/// Returns `(token [1] i32, logprob [1] f32)`.
fn sample_lp(
    logits: Tensor,
    vocab: u32,
    temperature: f32,
    top_p: f32,
    rng: Option<Channel>,
) -> (Tensor, Tensor) {
    let lp_dist = log_softmax(&logits);
    let token = match rng {
        None => reshape(reduce_argmax(&logits), [1]),
        Some(rng) => {
            let scaled = &logits / temperature;
            let masked = if top_p < 1.0 {
                let probs = softmax(&scaled);
                let keep = pivot_threshold(probs, cummass_le(top_p));
                let neg_inf = broadcast(f32::NEG_INFINITY, [vocab]);
                select(&keep, &scaled, &neg_inf)
            } else {
                scaled
            };
            let r = rng.take();
            let g = gumbel(&r, [vocab]);
            let r_next = &r + iota(2);
            rng.put(&r_next);
            reshape(reduce_argmax(masked + g), [1])
        }
    };
    let lp = scalar_gather(&lp_dist, cast(&token, dtype::u32));
    (token, lp)
}

struct RunResult {
    num_prompt_tokens: usize,
    token_ids: Vec<u32>,
    logprobs: Vec<f32>,
    finish_reason: String,
    cached_tokens: usize,
    saved_len: usize,
}

/// State binding over the two forward interfaces this inferlet runs on
/// (attention-only and hybrid), same shape as `text-completion-bench`.
trait BindState {
    fn bind_state<R, W>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, W>,
        rs: &[RsWorkingSet],
    ) -> ::std::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>;
}

impl BindState for inferlet::ptir::attention::ForwardPass {
    fn bind_state<R, W>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, W>,
        rs: &[RsWorkingSet],
    ) -> ::std::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        debug_assert!(rs.is_empty());
        self.attention(ws, geom)
    }
}

impl BindState for inferlet::ptir::hybrid::ForwardPass {
    fn bind_state<R, W>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, W>,
        rs: &[RsWorkingSet],
    ) -> ::std::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        self.attention(
            Some(KvBinding {
                working_set: ws,
                geometry: geom,
            }),
            rs,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )
    }
}

macro_rules! define_run_one {
    ($name:ident, $kind:ident) => {
        async fn $name(
            input: &Input,
            stop_tokens: &[u32],
            rendered: Option<&[u32]>,
        ) -> Result<RunResult> {
            use inferlet::ptir::$kind::{ForwardPass, submit_frame};

            let prompt_vec: Vec<u32> = if let Some(tokens) =
                rendered.or(input.prompt_tokens.as_deref())
            {
                tokens.to_vec()
            } else {
                let mut p = chat::system_user("You are a helpful assistant.", &input.prompt);
                p.extend(chat::cue());
                p
            };
            let num_prompt_tokens = prompt_vec.len();
            if prompt_vec.is_empty() {
                return Err("empty prompt".into());
            }
            if input.max_tokens == 0 {
                return Err("max_tokens must be >= 1".into());
            }

            let vocab = model::output_vocab_size();
            let temperature = input.temperature;
            let top_p = input.top_p;
            let sampled_rng =
                (temperature > 0.0).then(|| Channel::from([input.seed, 0u32]).named("rng"));
            let prefill_rng = (temperature > 0.0)
                .then(|| Channel::from([input.seed ^ 0x9e37_79b9, 0u32]).named("rng_p"));

            let page_size = kv_page_size();
            let n = prompt_vec.len() as u32;

            // ── KV snapshot resume (attention-only: the index API cannot carry
            // a hybrid's folded recurrent state, and a KV-only resume would
            // silently run its GDN layers cold).
            let kv_reuse_ok = model::pass_kind() == model::ForwardKind::Attention;
            let mut resumed_len: u32 = 0;
            let mut ws_opt: Option<WorkingSet> = None;
            if kv_reuse_ok && !input.saved_lens.is_empty() {
                let mut candidates: Vec<u32> = input
                    .saved_lens
                    .iter()
                    // Keep at least one suffix token: a prefill pass needs it,
                    // and it re-anchors the resumed KV to this exact prompt.
                    .map(|&l| l.min(n.saturating_sub(1)))
                    .filter(|&l| l > 0)
                    .collect();
                candidates.sort_unstable_by(|a, b| b.cmp(a));
                candidates.dedup();
                for len in candidates {
                    let key = index_key(&prompt_vec[..len as usize]);
                    match WorkingSet::from_index(&key) {
                        Ok(Some(shared)) => {
                            // Take-on-hit (the 0.4 session-cache discipline):
                            // this turn's save supersedes the boundary being
                            // resumed, and an index entry left behind pins its
                            // pages forever — 30+ cumulative turns of pinned
                            // boundaries exhausted the pool (pie_cuda_copy_kv
                            // status -5). ≤1 live snapshot per lineage. The
                            // looked-up working set stays valid past removal.
                            let _ = WorkingSet::remove_index(&key);
                            resumed_len = len;
                            ws_opt = Some(shared);
                            break;
                        }
                        Ok(None) => continue, // miss (stale hint or evicted)
                        Err(e) => return Err(format!("from_index: {e}").into()),
                    }
                }
            }
            let ws = ws_opt.unwrap_or_default();
            let max_pages = (n + input.max_tokens as u32 + 1).div_ceil(page_size);
            let reserve_to_tokens = |tokens: u32| -> std::result::Result<(), String> {
                let target = tokens.div_ceil(page_size).saturating_add(1).min(max_pages);
                let current = ws.page_len();
                if current < target {
                    ws.reserve(target - current)?;
                }
                Ok(())
            };
            reserve_to_tokens(n.max(1)).context("ws.reserve prompt")?;

            let prompt_i32: Vec<i32> = prompt_vec.iter().map(|&t| t as i32).collect();

            // Chunked prefill against the driver's per-launch token capacity;
            // only the last chunk carries the sampling epilogue. With a resumed
            // snapshot only the suffix [resumed_len..n) is prefilled — the
            // whole point of the snapshot.
            let spans: Vec<(u32, u32)> = prefill_chunks(n - resumed_len, input.prefill_chunk)
                .iter()
                .map(|&(b, e)| (b + resumed_len, e + resumed_len))
                .collect();
            let (last_base, last_end) = *spans.last().expect("prefill_chunks is non-empty");

            let toks_p = Channel::from(&prompt_i32[last_base as usize..last_end as usize])
                .named("toks_p");
            let embed_indptr_p =
                Channel::from([0u32, last_end - last_base]).named("embed_indptr_p");
            let positions_p = Channel::from_iter(last_base..last_end).named("positions_p");
            let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
            let page_indptr_p =
                Channel::from([0u32, n.div_ceil(page_size)]).named("page_indptr_p");
            let w_slot_p =
                Channel::from_iter((last_base..last_end).map(|p| p / page_size)).named("w_slot_p");
            let w_off_p =
                Channel::from_iter((last_base..last_end).map(|p| p % page_size)).named("w_off_p");

            // The prefill sample spends one of the max_tokens sampler activations.
            let budget = input.max_tokens - 1;

            let k = frame_size();
            let _ = k;
            let tok_in = Channel::new([1], dtype::i32).named("tok_in");
            let g0_ch = Channel::new([1], dtype::i32).named("g0");
            let lp0_ch = Channel::new([1], dtype::f32).named("lp0");
            let live_slots = live_slots();

            // Engine-sized run-ahead window (see text-completion-bench for the
            // full sizing rationale; this inferlet always uses the engine's own
            // `channel_capacity()` depth).
            let cap = channel_capacity();
            let (window_fires, out_capacity) = (cap - 1, cap + 7 * live_slots);
            let out = Channel::new([1], dtype::i32)
                .capacity(out_capacity as u32)
                .named("out");
            let lp_out = Channel::new([1], dtype::f32)
                .capacity(out_capacity as u32)
                .named("lp_out");

            let rs_ws: Vec<RsWorkingSet> = if model::pass_kind() != model::ForwardKind::Attention {
                vec![RsWorkingSet::new()]
            } else {
                Vec::new()
            };

            let pipe = Pipeline::new();

            // Leading prefill chunks: KV-extend only; epilogue sample is
            // mandatory at registration, so it goes to a drained throwaway.
            for &(base, end) in &spans[..spans.len() - 1] {
                let toks_c =
                    Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
                let embed_indptr_c = Channel::from([0u32, end - base]).named("embed_indptr_p");
                let positions_c = Channel::from_iter(base..end).named("positions_p");
                let pages_c = Channel::from_iter(0..max_pages).named("pages_p");
                let page_indptr_c =
                    Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_p");
                let w_slot_c =
                    Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_p");
                let w_off_c =
                    Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_p");
                let kv_len_c = Channel::from([end]).named("kv_len_p");
                let fwd_c = ForwardPass::new();
                fwd_c.embed(&toks_c, &embed_indptr_c)?;
                fwd_c
                    .bind_state(
                        &ws,
                        KvGeometry {
                            readable_pages: ..,
                            writable_pages: ..,
                            kv_len: &kv_len_c,
                            pages: &pages_c,
                            page_indptr: &page_indptr_c,
                            w_slot: &w_slot_c,
                            w_off: &w_off_c,
                            positions: &positions_c,
                            mask: None,
                        },
                        &rs_ws,
                    )
                    .with_context(|| format!("bind prefill chunk @{base}"))?;
                let drop_tok_c = Channel::new([1], dtype::i32).named("drop_tok_c");
                let drop_sink = drop_tok_c.clone();
                // Greedy, rng-less sample: the output is discarded, and a
                // seeded rng channel stages exactly ONE cell consumed by the
                // first pass instance that fires — sharing prefill_rng across
                // several chunk passes fails the second chunk's instantiation
                // with MissingSeed ("seeded but no seed was put before the
                // first fire").
                fwd_c.epilogue(move || {
                    let (t, _lp) = sample_lp(intrinsics::logits(), vocab, 0.0, 1.0, None);
                    drop_sink.put(&t);
                });
                fwd_c
                    .submit(&pipe)
                    .with_context(|| format!("prefill chunk submit @{base}"))?;
                drop_tok_c
                    .take_host::<i32>()
                    .await
                    .with_context(|| format!("drain prefill chunk @{base}"))?;
            }

            let fwd_p = ForwardPass::new();
            fwd_p.embed(&toks_p, &embed_indptr_p)?;
            let kv_len_p = Channel::from([n]).named("kv_len_p");
            fwd_p
                .bind_state(
                    &ws,
                    KvGeometry {
                        readable_pages: ..,
                        writable_pages: ..,
                        kv_len: &kv_len_p,
                        pages: &pages_p,
                        page_indptr: &page_indptr_p,
                        w_slot: &w_slot_p,
                        w_off: &w_off_p,
                        positions: &positions_p,
                        mask: None,
                    },
                    &rs_ws,
                )
                .context("bind prefill state")?;
            fwd_p.epilogue(move || {
                let (t, lp) =
                    sample_lp(intrinsics::logits(), vocab, temperature, top_p, prefill_rng);
                tok_in.put(&t);
                g0_ch.put(&t);
                lp0_ch.put(&lp);
            });

            // Decode pass (1-wide, device loop-carried), built before the
            // prefill submits.
            let fwd_d: Option<ForwardPass> = if budget > 0 {
                let fwd = ForwardPass::new();
                let embed_indptr = Channel::from([0u32, 1u32]).named("embed_indptr");
                let positions = Channel::from([n]).named("positions");
                let pages = Channel::from_iter(0..max_pages).named("pages");
                let page_indptr =
                    Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
                let w_slot = Channel::from([n / page_size]).named("w_slot");
                let w_off = Channel::from([n % page_size]).named("w_off");
                fwd.embed(&tok_in, &embed_indptr)?;
                let kv_len = Channel::from([n + 1]).named("kv_len");
                fwd.bind_state(
                    &ws,
                    KvGeometry {
                        readable_pages: ..,
                        writable_pages: (n / page_size)..,
                        kv_len: &kv_len,
                        pages: &pages,
                        page_indptr: &page_indptr,
                        w_slot: &w_slot,
                        w_off: &w_off,
                        positions: &positions,
                        mask: None,
                    },
                    &rs_ws,
                )
                .context("bind decode state")?;
                let out_d = out.clone();
                let lp_out_d = lp_out.clone();
                fwd.epilogue(move || {
                    let length = kv_len.take();
                    let (t, lp) =
                        sample_lp(intrinsics::logits(), vocab, temperature, top_p, sampled_rng);
                    let next_length = &length + 1u32;
                    let page_count = next_length.div_ceil(page_size);
                    tok_in.put(&t);
                    kv_len.put(&next_length);
                    positions.put(&length);
                    w_slot.put(&length / page_size);
                    w_off.put(&length % page_size);
                    page_indptr.put(indptr(1, &page_count));
                    out_d.put(&t);
                    lp_out_d.put(&lp);
                });
                Some(fwd)
            } else {
                None
            };

            // First frame: prefill in slot 0, then up to live_slots-1 decodes.
            let first_decodes = budget.min(live_slots - 1);
            reserve_to_tokens(n + first_decodes as u32 + 1).context("reserve first frame")?;
            let mut first_slots: Vec<Option<&ForwardPass>> = Vec::with_capacity(live_slots);
            first_slots.push(Some(&fwd_p));
            for _ in 0..first_decodes {
                first_slots.push(Some(fwd_d.as_ref().expect("decode pass exists")));
            }
            submit_frame(&pipe, &first_slots).context("first frame submit")?;
            let mut submitted = first_decodes;

            // Run-ahead discipline, same rule as text-completion-bench.
            let submit_ahead =
                |mut submitted: usize, drained: usize| -> std::result::Result<usize, String> {
                    while submitted < budget {
                        let s = (budget - submitted).min(live_slots);
                        if submitted - drained + s > window_fires {
                            break;
                        }
                        reserve_to_tokens(n + (submitted + s) as u32 + 1)
                            .context("reserve decode frame")?;
                        let fwd = fwd_d.as_ref().expect("decode pass exists while budget > 0");
                        let slots: Vec<Option<&ForwardPass>> =
                            (0..s).map(|_| Some(fwd)).collect();
                        submit_frame(&pipe, &slots).context("decode frame submit")?;
                        submitted += s;
                    }
                    Ok(submitted)
                };
            submitted = submit_ahead(submitted, 0)?;

            // ── HOST DRAIN ──
            let g0 = g0_ch.take_host::<i32>().await?;
            let lp0 = lp0_ch
                .take_host::<Vec<f32>>()
                .await
                .context("drain lp0")?;
            let lp0 = *lp0.first().ok_or("lp0.take: empty tensor")?;

            let mut token_ids: Vec<u32> = Vec::with_capacity(input.max_tokens);
            let mut logprobs: Vec<f32> = Vec::with_capacity(input.max_tokens);
            token_ids.push(g0 as u32);
            logprobs.push(lp0);
            let mut stopped = stop_tokens.contains(&(g0 as u32));

            let mut closed = false;
            if stopped || submitted >= budget {
                pipe.close();
                closed = true;
            }
            let mut taken = 0usize;
            while taken < submitted {
                let t = out.take_host::<Vec<i32>>().await?;
                let lp = lp_out.take_host::<Vec<f32>>().await.context("drain lp")?;
                taken += 1;
                let Some(&t0) = t.first() else {
                    return Err("out.take: empty tensor".into());
                };
                let Some(&lp0) = lp.first() else {
                    return Err("lp_out.take: empty tensor".into());
                };
                if stopped {
                    continue; // fires already staged when the stop landed
                }
                token_ids.push(t0 as u32);
                logprobs.push(lp0);
                if stop_tokens.contains(&(t0 as u32)) {
                    stopped = true;
                    if !closed {
                        pipe.close();
                        closed = true;
                    }
                    continue;
                }
                submitted = submit_ahead(submitted, taken)?;
                if !closed && submitted >= budget {
                    pipe.close();
                    closed = true;
                }
            }
            if !closed {
                pipe.close();
            }

            let finish_reason = if stopped { "stop" } else { "length" }.to_string();

            // ── KV snapshot save at the completion boundary. KV exists for the
            // prompt plus every generated token except the last (a sampled
            // token's KV is only written when it is embedded by the next fire),
            // so the boundary is prompt + token_ids[..len-1]. Positions beyond
            // it may hold garbage from run-ahead fires that landed after the
            // stop — harmless, a resume attends only up to its kv_len.
            let mut saved_len = 0usize;
            if kv_reuse_ok && input.save_kv && !token_ids.is_empty() {
                let mut seq = prompt_vec.clone();
                seq.extend_from_slice(&token_ids[..token_ids.len() - 1]);
                let key = index_key(&seq);
                // update_index requires a settled working set; the last fires'
                // writes can still be in flight just after the final drain.
                for _ in 0..100 {
                    match ws.update_index(&key) {
                        Ok(()) => {
                            saved_len = seq.len();
                            break;
                        }
                        Err(e) if e.contains("in flight") => {
                            std::thread::sleep(std::time::Duration::from_millis(10));
                        }
                        Err(e) => {
                            // Non-fatal by design: the turn's result is intact,
                            // the next turn just rebuilds.
                            eprintln!("rl-rollout: update_index failed: {e}");
                            break;
                        }
                    }
                }
            }

            Ok(RunResult {
                num_prompt_tokens,
                token_ids,
                logprobs,
                finish_reason,
                cached_tokens: resumed_len as usize,
                saved_len,
            })
        }
    };
}

define_run_one!(run_one_attention, attention);
define_run_one!(run_one_hybrid, hybrid);

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    // Pure request/response: no streaming, no external effects — safe for the
    // planner to restart under KV pressure instead of failing the request.
    inferlet::runtime::declare_restartable();
    let _ = session::receive; // (session unused in v0; keeps the import honest)

    let stop_tokens: Vec<u32> = if input.ignore_eos {
        Vec::new()
    } else {
        chat::stop_tokens()
    };

    let rendered: Option<Vec<u32>> = match &input.messages {
        Some(msgs) => Some(render_messages(msgs)?),
        None => None,
    };

    let result = match model::pass_kind() {
        model::ForwardKind::Attention => {
            run_one_attention(&input, &stop_tokens, rendered.as_deref()).await?
        }
        model::ForwardKind::Hybrid => {
            run_one_hybrid(&input, &stop_tokens, rendered.as_deref()).await?
        }
        model::ForwardKind::Recurrent => {
            return Err("rl-rollout has no recurrent-only path".to_string().into());
        }
    };

    let text = if input.return_text {
        let end = result.token_ids.len()
            - usize::from(result.finish_reason == "stop" && !result.token_ids.is_empty());
        model::decode(&result.token_ids[..end]).unwrap_or_default()
    } else {
        String::new()
    };

    Ok(Output {
        num_prompt_tokens: result.num_prompt_tokens,
        num_output_tokens: result.token_ids.len(),
        token_ids: result.token_ids,
        logprobs: result.logprobs,
        finish_reason: result.finish_reason,
        cached_tokens: result.cached_tokens,
        saved_len: result.saved_len,
        prompt_token_ids: rendered.unwrap_or_default(),
        text,
    })
}
