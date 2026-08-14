//! **Is a grafted prefill the honest chunked continuation?** At serving scale,
//! on the real driver and the real model.
//!
//! The per-request prefix cache in `inferlets/chat-completions` makes a turn
//! read KV pages it did not write. On a ~9.2k-token opencode-shaped prompt that
//! turn agreed with the cold turn on 3 of 6 prompts at temperature 0 — and a
//! disagreement there is ambiguous in the worst way, because TWO very different
//! things produce it:
//!
//! - a near-tie argmax flipping under different reduction order. Benign. A cold
//!   prefill chunked at 2048 already disagrees with one chunked at 1024 on 1 of
//!   6 prompts, with no cache involved at all.
//! - a graft that lands on a prefix ending in the wrong place. Silent, and it
//!   still decodes fluent text.
//!
//! Byte-comparing cold against warm cannot separate them, because those two
//! runs differ in BOTH ways at once: the cold run computes the tail inside
//! wide even chunks, the warm run computes it as one narrow fire.
//!
//! This probe removes the second difference and keeps only the question:
//!
//! ```text
//!   HONEST   one working set: chunk 0..cut, then fire cut..n, then decode.
//!            Parks pages 0..cut/page under KEY.
//!   GRAFT    from_index(KEY), reserve, fire cut..n, then decode.
//! ```
//!
//! Both compute `cut..n` with IDENTICAL fire shapes over KV for `0..cut` that
//! is not merely equivalent but the same bytes — the graft reads the very pages
//! the honest run wrote. Every input to the two computations is equal, so if
//! the graft's geometry (positions, write descriptors, page CSR, readable and
//! writable declarations) is right, the outputs must be EQUAL, not merely
//! close. Any difference is the geometry, and nothing else.
//!
//! FULL is reported alongside as the floor: a plain cold prefill of `0..n` in
//! even chunks, which is what the serving arm does on a miss. HONEST vs FULL is
//! the chunk-shape effect on its own, with no cache in the picture, and it is
//! the number that says how much cold-vs-warm disagreement was never about the
//! cache.
//!
//! Comparison is over a 32-token greedy continuation, not one argmax: a single
//! token agrees by luck far too often to mean anything.

use inferlet::Result;
use inferlet::ptir::attention::prelude::*;

const N: u32 = 9216;
const DECODE: usize = 32;
const KEY: &[u8] = b"apc-graft-probe/honest-prefix";
/// A prefix over DIFFERENT text, parked at the same cut. The negative control
/// grafts this turn's tail onto it: legal geometry in every respect, wrong
/// content — exactly what an address collision or a mis-scoped key would do.
const KEY_OTHER: &[u8] = b"apc-graft-probe/other-prefix";

/// Real text, tokenized by the served model.
///
/// The first version of this probe used pseudo-random token ids, and all three
/// arms agreed on all 33 tokens — including FULL, which the six-prompt control
/// run says should disagree sometimes. The reason is the failure mode this
/// whole probe exists to avoid: on gibberish the model emits filler (the same
/// id repeatedly), and a run whose output does not depend on its context CANNOT
/// detect a graft that reads the wrong context. Agreement was free.
///
/// Text the model can actually follow makes every token a function of the
/// prefix, which is the only condition under which equality means anything.
fn enc(s: &str) -> Vec<i32> {
    model::encode(s).into_iter().map(|t| t as i32).collect()
}

fn notes(seed: u32) -> String {
    let mut text = String::new();
    for i in seed..seed + 400 {
        text.push_str(&format!(
            "Repository note {i}: module pkg/mod_{i}.py defines helper_{i}(x), \
             which returns x * {i} and is covered by tests/test_mod_{i}.py.\n"
        ));
    }
    text
}

/// `cut` tokens of head carrying a unique code, then a shared tail that asks
/// for it. Returns the whole stream.
///
/// The code has to sit in the PARKED half and the question in the grafted half,
/// or the probe measures nothing — which is not a hypothetical. Two earlier
/// versions of this file were insensitive:
///
/// - pseudo-random token ids: the model emitted filler, so all four arms agreed
///   because no arm's output depended on any arm's input;
/// - self-similar repository notes: the continuation after 4.6k tokens of notes
///   is more notes, so a prefix built from DIFFERENT notes produced the same
///   33 tokens. Wrong context, identical answer, `PROBE_INSENSITIVE`.
///
/// Both looked like a clean pass. A probe that cannot fail is not evidence, so
/// the content is now chosen so that the only way to answer is to read the
/// parked half.
fn stream(n: u32, cut: u32, code: &str) -> Vec<i32> {
    let mut head = enc(&format!(
        "IMPORTANT. The access code for this repository is {code}. \
         Write it down: the access code is {code}.\n"
    ));
    let filler = enc(&notes(0));
    while head.len() < cut as usize {
        head.extend(filler.iter());
    }
    head.truncate(cut as usize);

    // Identical in every stream, so the grafted fire runs the same tokens
    // whichever head is behind it.
    let question = enc("\n\nQuestion: what is the access code?\nAnswer: the access code is");
    let mut tail = Vec::new();
    let filler2 = enc(&notes(900));
    while tail.len() + question.len() < (n - cut) as usize {
        tail.extend(filler2.iter());
    }
    tail.truncate((n - cut) as usize - question.len());
    tail.extend(question);

    head.extend(tail);
    head
}

/// One prefill fire over `base..end`, reading everything and writing only at or
/// past `writable_from`. Returns the greedy read-out token for row `end-1`.
///
/// Mirrors `inferlets/chat-completions/src/engine.rs` exactly — this is the
/// geometry under test, so it is written the same way rather than a tidier way.
async fn fire(
    ws: &WorkingSet,
    pipe: &Pipeline,
    toks: &[i32],
    base: u32,
    end: u32,
    page: u32,
    pool_ids: &[u32],
    writable_from: u32,
) -> std::result::Result<i32, String> {
    let len = end - base;
    let t = Channel::from(&toks[base as usize..end as usize]).named("toks");
    let embed_indptr = Channel::from([0u32, len]).named("embed_indptr");
    let positions = Channel::from_iter(base..end).named("positions");
    let w_slot = Channel::from(
        (base..end)
            .map(|c| pool_ids[(c / page) as usize])
            .collect::<Vec<u32>>(),
    )
    .named("w_slot");
    let w_off = Channel::from((base..end).map(|c| c % page).collect::<Vec<u32>>()).named("w_off");
    let kv_len = Channel::from([end]).named("kv_len");
    let pages = Channel::from(pool_ids.to_vec()).named("pages");
    let page_indptr = Channel::from([0u32, end.div_ceil(page)]).named("page_indptr");
    let out = Channel::new([1], dtype::i32).named("out");

    let fwd: ForwardPass = ForwardPass::new();
    fwd.embed(&t, &embed_indptr)?;
    fwd.attention(
        ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: writable_from..,
            kv_len: &kv_len,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &positions,
            mask: None,
        },
    )?;
    fwd.epilogue(move || out.put(&reduce_argmax(intrinsics::logits())));
    fwd.submit(pipe)
        .with_context(|| format!("submit {base}..{end}"))?;
    out.take_host::<i32>()
        .await
        .with_context(|| format!("take {base}..{end}"))
}

/// `DECODE` greedy tokens starting from `first` at position `n`. Sequential
/// (submit, take, submit) rather than run-ahead: this probe compares token
/// SEQUENCES, and speculation would leave fires executed past the comparison.
async fn decode(
    ws: &WorkingSet,
    pipe: &Pipeline,
    first: i32,
    n: u32,
    page: u32,
    pool_ids: &[u32],
) -> std::result::Result<Vec<i32>, String> {
    let mut got = vec![first];
    let mut tok = first;
    for k in 0..DECODE as u32 {
        let pos = n + k;
        let one = [tok];
        // A 1-token fire is the same geometry with base=pos, end=pos+1; the
        // token comes from the previous step rather than the prompt.
        let t = Channel::from(&one[..]).named("d_tok");
        let embed_indptr = Channel::from([0u32, 1]).named("d_indptr");
        let positions = Channel::from([pos]).named("d_pos");
        let w_slot = Channel::from([pool_ids[(pos / page) as usize]]).named("d_wslot");
        let w_off = Channel::from([pos % page]).named("d_woff");
        let kv_len = Channel::from([pos + 1]).named("d_klen");
        let pages = Channel::from(pool_ids.to_vec()).named("d_pages");
        let page_indptr = Channel::from([0u32, (pos + 1).div_ceil(page)]).named("d_pidx");
        let out = Channel::new([1], dtype::i32).named("d_out");

        let fwd: ForwardPass = ForwardPass::new();
        fwd.embed(&t, &embed_indptr)?;
        fwd.attention(
            ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: (pos / page)..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        )?;
        fwd.epilogue(move || out.put(&reduce_argmax(intrinsics::logits())));
        fwd.submit(pipe)
            .with_context(|| format!("decode submit @{pos}"))?;
        tok = out
            .take_host::<i32>()
            .await
            .with_context(|| format!("decode take @{pos}"))?;
        got.push(tok);
    }
    Ok(got)
}

fn pool(ws: &WorkingSet, want_pages: u32) -> std::result::Result<Vec<u32>, String> {
    let have = ws.page_len();
    let mut ids: Vec<u32> = (0..have).collect();
    if want_pages > have {
        ids.extend_from_slice(ws.reserve(want_pages - have)?.ids());
    }
    Ok(ids)
}

/// Prefill `0..cut` in chunks, fire `cut..n`, decode. Park pages `0..cut/page`.
async fn honest(
    toks: &[i32],
    n: u32,
    cut: u32,
    page: u32,
) -> std::result::Result<Vec<i32>, String> {
    let ws = WorkingSet::new();
    let ids = pool(&ws, (n + DECODE as u32 + 2).div_ceil(page))?;
    let pipe = Pipeline::new();
    for (b, e) in prefill_chunks(cut, None) {
        fire(&ws, &pipe, toks, b, e, page, &ids, 0).await?;
    }
    // The tail, chunked exactly as the graft will chunk it.
    let mut g = 0;
    for (b, e) in prefill_chunks(n - cut, None) {
        g = fire(&ws, &pipe, toks, b + cut, e + cut, page, &ids, cut / page).await?;
    }
    let out = decode(&ws, &pipe, g, n, page, &ids).await?;
    // Parked LAST: `slice` makes pages shared, and a later fire writing near
    // them would need a copy-on-write KV copy, which the Metal driver refuses
    // for any checkpoint that is not a GDN hybrid (`copy_kv_impl`).
    ws.slice(&pipe, 0, cut / page)?.update_index(KEY)?;
    pipe.close();
    Ok(out)
}

/// Prefill `0..cut` of a DIFFERENT token stream and park it under `key`.
/// Nothing is read out — this exists only to give the negative control a prefix
/// that is structurally perfect and semantically wrong.
async fn park_other(
    toks: &[i32],
    cut: u32,
    page: u32,
    key: &[u8],
) -> std::result::Result<(), String> {
    let ws = WorkingSet::new();
    let ids = pool(&ws, cut.div_ceil(page))?;
    let pipe = Pipeline::new();
    for (b, e) in prefill_chunks(cut, None) {
        fire(&ws, &pipe, toks, b, e, page, &ids, 0).await?;
    }
    ws.slice(&pipe, 0, cut / page)?.update_index(key)?;
    pipe.close();
    Ok(())
}

/// Load the parked prefix, fire `cut..n` on top of it, decode.
async fn graft(
    toks: &[i32],
    n: u32,
    cut: u32,
    page: u32,
    key: &[u8],
) -> std::result::Result<Vec<i32>, String> {
    let ws = WorkingSet::from_index(key)?.ok_or("graft: parked prefix not found")?;
    if ws.page_len() != cut / page {
        return Err(format!(
            "graft: parked set holds {} page(s), expected {}",
            ws.page_len(),
            cut / page
        ));
    }
    let ids = pool(&ws, (n + DECODE as u32 + 2).div_ceil(page))?;
    let pipe = Pipeline::new();
    let mut g = 0;
    for (b, e) in prefill_chunks(n - cut, None) {
        g = fire(&ws, &pipe, toks, b + cut, e + cut, page, &ids, cut / page).await?;
    }
    let out = decode(&ws, &pipe, g, n, page, &ids).await?;
    pipe.close();
    Ok(out)
}

/// Cold prefill of the whole prompt in even chunks — the serving arm's miss
/// path. Its chunk boundaries do not line up with `cut`, which is the point.
async fn full(toks: &[i32], n: u32, page: u32) -> std::result::Result<Vec<i32>, String> {
    let ws = WorkingSet::new();
    let ids = pool(&ws, (n + DECODE as u32 + 2).div_ceil(page))?;
    let pipe = Pipeline::new();
    let mut g = 0;
    for (b, e) in prefill_chunks(n, None) {
        g = fire(&ws, &pipe, toks, b, e, page, &ids, 0).await?;
    }
    let out = decode(&ws, &pipe, g, n, page, &ids).await?;
    pipe.close();
    Ok(out)
}

fn agree_upto(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let page = kv_page_size();
    let n = (N / page) * page; // whole pages, so `cut` arithmetic is exact
    let cut = ((n / 2) / page) * page;
    // Same tail, different parked code — so the ONLY way an arm can answer is
    // by reading the half it did not compute.
    let toks = stream(n, cut, "ALPHA-7731");
    let other = stream(n, cut, "BRAVO-2244");
    println!(
        "[apc-graft] n={n} cut={cut} page={page} chunk_cap={} tail_chunks={}",
        max_embed_length(),
        prefill_chunks(n - cut, None).len()
    );

    let h = honest(&toks, n, cut, page).await?;
    let g = graft(&toks, n, cut, page, KEY).await?;
    let f = full(&toks, n, page).await?;

    // NEGATIVE CONTROL. This turn's tail grafted onto a prefix built from
    // DIFFERENT text: every position, write descriptor, page CSR and page
    // declaration is correct, and only the content behind the cut is wrong.
    //
    // It has to be content rather than geometry. The first version skewed the
    // positions by one page, and the driver REFUSED the launch outright — "a
    // token at position 6144 has no page in its request's list" — which is
    // reassuring about over-long resumes but tests the driver, not this probe.
    // Wrong content cannot be refused by anything: it is the failure a prefix
    // cache actually has, and if the probe cannot see it then `GRAFT_EXACT`
    // above means only that the model ignored its context.
    park_other(&other, cut, page, KEY_OTHER).await?;
    let w = graft(&toks, n, cut, page, KEY_OTHER).await?;

    let hg = agree_upto(&h, &g);
    let hf = agree_upto(&h, &f);
    let hw = agree_upto(&h, &w);
    let say = |v: &[i32]| {
        model::decode(&v.iter().map(|&t| t as u32).collect::<Vec<u32>>())
            .unwrap_or_default()
            .replace('\n', " ")
    };
    println!("[apc-graft] honest {:?}", say(&h));
    println!("[apc-graft] graft  {:?}", say(&g));
    println!("[apc-graft] full   {:?}", say(&f));
    println!("[apc-graft] wrong  {:?}", say(&w));

    // The claim: identical inputs, identical fire shapes => identical output.
    // Not "close" — equal. The wrong-prefix arm is here to prove that sentence
    // has teeth rather than assume it.
    let verdict = if hw == h.len() {
        "PROBE_INSENSITIVE"
    } else if hg == h.len() {
        "GRAFT_EXACT"
    } else {
        "GRAFT_DIVERGED"
    };
    let result = format!(
        "APC_GRAFT_PROBE {verdict} n={n} cut={cut} tokens={} \
         graft_agrees={hg} full_agrees={hf} wrong_prefix_agrees={hw}",
        h.len()
    );
    println!("{result}");
    Ok(result)
}
