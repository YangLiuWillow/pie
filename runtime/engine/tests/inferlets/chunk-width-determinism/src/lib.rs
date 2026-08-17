//! **Does the WIDTH of a prefill fire change the logits it produces?**
//!
//! ## The question, and why it is not academic
//!
//! Chunking a prompt is supposed to be an implementation detail: the same
//! tokens, the same weights, the same KV, split across a different number of
//! fires. If the resulting distribution depends on how the prompt was sliced,
//! then two callers that agree on everything a user can see still disagree on
//! the answer, and a cache that was filled at one chunk width is not
//! interchangeable with one filled at another.
//!
//! A sibling investigation on CUDA reports exactly that, with a pattern that
//! resists the obvious explanations: at an 8409-token prompt, NINE chunks is
//! bit-identical to one-shot while TEN differs; a 516-token chunk is exact and
//! an 841-token chunk is not. Not monotonic in chunk count, not a size
//! threshold — discrete and shape-keyed.
//!
//! ## The hypothesis this tests
//!
//! pie selects attention kernels on RUNTIME SHAPE, and we have already proved
//! on Metal that different kernels give different tokens (`_u4` versus the plain
//! head-sharing kernel diverges in generated text, including below the context
//! gate meant to make it inert). The gates on this driver key on the row count
//! of the fire:
//!
//!   * `sdpa_split_this_fire`  — `rows == 1` takes split-K
//!   * `sdpa_should_tile`      — a fire earns the tiled kernel by filling a
//!                               32-row tile
//!   * `sdpa_nax_min_rows`     — the neural-accelerator prefill path has its own
//!                               minimum width
//!
//! **Chunk width IS rows per fire.** So different widths land on different
//! kernel instantiations, and a discrete, shape-keyed, non-monotonic pattern is
//! precisely what that would produce.
//!
//! ## That hypothesis was TESTED AND REFUTED, and the answer is more general
//!
//! `PIE_METAL_SDPA_TRACE=1` across nine widths reports IDENTICAL kernel
//! selection in every arm -- every prefill fire takes NAX, since `rows >= 32`
//! holds even for the narrow tail chunks -- while eight of the nine produce
//! distinct results. Kernel choice is not the variable.
//!
//! The mechanism is REDUCTION ORDER. For a token at position `p` in chunk
//! `[b,e)`, attention reads keys below `b` from the paged cache and keys in
//! `[b,p]` from this fire's own freshly written KV. Moving the boundary moves
//! that split, the online softmax accumulates in a different order, and float
//! addition is not associative. Same code, different partition, different bits.
//!
//! That is stronger than the hypothesis it replaces: ANY chunking perturbs
//! prefill numerics on ANY backend, whatever kernel runs. It explains the CUDA
//! observation without an autotuner, and it predicts the non-monotonicity --
//! there is no reason for one partition's rounding to order sensibly against
//! another's.
//!
//! ## What is held fixed
//!
//! The prompt tokens, the page pool, the working set, and the total context are
//! identical across arms. ONLY the fire boundaries move. The reported value is
//! the LAST chunk's next-token distribution — the one a caller would actually
//! consume — because intermediate chunks' logits are mid-prompt and discarded by
//! any real serving path.
//!
//! Raw bits, not decimals: this is a bitwise question and `{:.6}` would round
//! two different floats onto the same string.
//!
//! ## Arguments
//!
//! `-- prompt=8192,chunk=1024`, where `chunk=0` means one shot. Unparseable
//! input fails rather than defaulting, because an arm that silently ran the
//! default width would report agreement between two identical configurations.

use inferlet::Result;
use inferlet::ptir::attention::prelude::*;
use inferlet::model as wit_model;

const DEFAULT_PROMPT: u32 = 8192;

fn positional(input: &str) -> &str {
    const KEY: &str = "\"_positional\":[\"";
    match input.find(KEY) {
        Some(at) => {
            let rest = &input[at + KEY.len()..];
            rest.find('"').map_or(rest, |end| &rest[..end])
        }
        None if input.trim_start().starts_with('{') => "",
        None => input.trim(),
    }
}

fn config(input: &str) -> Result<(u32, u32)> {
    let (mut prompt, mut chunk) = (DEFAULT_PROMPT, 0u32);
    for field in input.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (k, v) = field
            .split_once('=')
            .ok_or_else(|| format!("bad argument {field:?}: want key=value"))?;
        match k {
            "prompt" => prompt = v.parse().map_err(|_| format!("bad prompt {v:?}"))?,
            "chunk" => chunk = v.parse().map_err(|_| format!("bad chunk {v:?}"))?,
            _ => return Err(format!("unknown argument {k:?}")),
        }
    }
    Ok((prompt, chunk))
}

/// Deterministic filler, so every arm prefills byte-identical tokens.
fn notes() -> String {
    let mut text = String::new();
    for i in 0..1200 {
        text.push_str(&format!(
            "Repository note {i}: module pkg/mod_{i}.py defines helper_{i}(x), \
             which returns x * {i} and is covered by tests/test_mod_{i}.py.\n"
        ));
    }
    text
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let (want_prompt, chunk) = config(positional(&_input))?;
    let ws = WorkingSet::new();
    let page = kv_page_size();

    let mut toks: Vec<i32> = wit_model::encode(&notes())
        .into_iter()
        .map(|t| t as i32)
        .collect();
    if toks.len() < want_prompt as usize {
        return Err(format!(
            "filler is {} tokens, need {want_prompt}; widen `notes()`",
            toks.len()
        ));
    }
    toks.truncate(want_prompt as usize);
    let n = want_prompt;

    let max_pages = (n + 1).div_ceil(page);
    let have = ws.page_len();
    if max_pages > have {
        ws.reserve(max_pages - have).context("ws.reserve")?;
    }
    let pool: Vec<u32> = (0..max_pages).collect();
    let pipe = Pipeline::new();

    // `chunk == 0` is one shot; otherwise fixed-width slices with a short tail.
    let bounds: Vec<(u32, u32)> = if chunk == 0 {
        vec![(0, n)]
    } else {
        let mut v = Vec::new();
        let mut b = 0;
        while b < n {
            v.push((b, (b + chunk).min(n)));
            b += chunk;
        }
        v
    };
    println!(
        "[cwd] prompt={n} chunk={chunk} fires={} widths={:?}",
        bounds.len(),
        bounds.iter().map(|(b, e)| e - b).collect::<Vec<u32>>()
    );

    let last = bounds.len() - 1;
    let mut result = String::new();
    for (i, &(base, end)) in bounds.iter().enumerate() {
        let len = end - base;
        let t = Channel::from(&toks[base as usize..end as usize]).named("p_toks");
        let indptr = Channel::from([0u32, len]).named("p_indptr");
        let positions = Channel::from_iter(base..end).named("p_pos");
        let w_slot = Channel::from(
            (base..end)
                .map(|c| pool[(c / page) as usize])
                .collect::<Vec<u32>>(),
        )
        .named("p_wslot");
        let w_off =
            Channel::from((base..end).map(|c| c % page).collect::<Vec<u32>>()).named("p_woff");
        let kv_len = Channel::from([end]).named("p_klen");
        let pages = Channel::from(pool.clone()).named("p_pages");
        let page_indptr = Channel::from([0u32, end.div_ceil(page)]).named("p_pidx");
        let out_tok = Channel::new([1], dtype::i32).named("p_tok");
        let out_lp = Channel::new([1], dtype::f32).named("p_lp");
        let out_h = Channel::new([1], dtype::f32).named("p_h");

        let fwd = ForwardPass::new();
        fwd.embed(&t, &indptr)?;
        fwd.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: (base / page)..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        )?;
        // Every chunk publishes, not only the last: the values are cheap and an
        // arm that diverges EARLY is a different finding from one that diverges
        // only at the end, which a last-chunk-only probe could not tell apart.
        fwd.epilogue(move || {
            let logits = intrinsics::logits();
            let lp = log_softmax(&logits);
            out_tok.put(&reduce_argmax(&logits));
            out_lp.put(&reduce_max(&lp));
            out_h.put(&entropy(softmax(&logits)));
        });
        fwd.submit(&pipe)
            .with_context(|| format!("prefill submit {base}..{end}"))?;
        let tok = out_tok.take_host::<Vec<i32>>().await?;
        let lp = out_lp.take_host::<Vec<f32>>().await?;
        let h = out_h.take_host::<Vec<f32>>().await?;
        let (tok, lp, h) = (
            *tok.first().ok_or("token empty")?,
            *lp.first().ok_or("logprob empty")?,
            *h.first().ok_or("entropy empty")?,
        );
        let line = format!(
            "[cwd] fire={i} span={base}..{end} rows={len} tok={tok} \
             lp_bits={:08x} h_bits={:08x}{}",
            lp.to_bits(),
            h.to_bits(),
            if i == last { "  <- FINAL" } else { "" }
        );
        println!("{line}");
        if i == last {
            result = format!("tok={tok} lp_bits={:08x} h_bits={:08x}", lp.to_bits(), h.to_bits());
        }
    }
    pipe.close();
    println!("[cwd] final {result}");
    Ok(result)
}
