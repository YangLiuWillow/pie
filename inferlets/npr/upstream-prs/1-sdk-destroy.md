## The bug

`Context::destroy` (`sdk/rust/inferlet/src/context.rs`) consumes `self` and calls the host's `destroy`, which deletes the context's entry from the component resource table (`runtime/src/api/context.rs`, `HostContext::destroy` → `self.ctx().table.delete(this)?`).

When `destroy` returns, the SDK's `RawContext` handle is still dropped normally. wit-bindgen's resource drop invokes the host `drop` handler, which begins with `self.ctx().table.get(&this)?` — on an entry that `destroy` just deleted. The `get` fails, `drop` returns `Err`, and the instance traps.

## The failure it causes

Any inferlet that calls `Context::destroy` crashes on the very next instruction after the destroy. There is no partial-failure mode: the guest instance traps. This makes explicit context lifetime management unusable from the Rust SDK — which matters for any workload that creates and releases many contexts (fork/join search, tree-of-thought, parallel branch decoding), where waiting for process teardown to reclaim contexts is not an option.

## The fix

Destructure the wrapper, destroy the host resource, then `mem::forget` the handle so the resource drop never fires:

```rust
pub fn destroy(self) {
    let Context { inner, .. } = self;
    inner.destroy();
    std::mem::forget(inner);
}
```

## Relationship to #484 — please read before merging

pie-project/pie#484 ("Fix borrowed context destruction at the component boundary") fixes the same defect from the **host** side: it drops the `table.delete(this)?` from `HostContext::destroy` and leaves handle deletion to the ordinary resource drop. That PR was merged into `tts-arena/main`, a branch that no longer exists; **`main` still carries the bug** (`HostContext::destroy` on `main` still calls `table.delete(this)?`).

The two fixes are alternatives and must **not** both be applied:

- With this SDK-side fix alone (the state of `main` today): `destroy` deletes the entry, and the SDK forgets the handle so nothing double-frees. Correct.
- With #484's host-side fix alone: `destroy` no longer deletes, and the ordinary drop deletes. Correct.
- With both: the entry is never deleted — `mem::forget` suppresses the only remaining deleter, leaking a resource-table slot per destroyed context.

I have no preference; #484 is arguably the better layering, and cherry-picking it onto `main` would supersede this PR entirely. I am opening this because the bug is live on `main` and the host-side fix never landed there. Whichever way you go, please make sure exactly one of them ends up on `main`.

## Verification

- Read against current `main`: `HostContext::destroy` deletes the table entry; `HostContext::drop` starts with `table.get(&this)?`. The trap is reachable as described.
- `cargo check --target wasm32-wasip2` in `sdk/rust/inferlet`: clean.
- Behaviourally: found while building a fork/join reasoning inferlet that creates and destroys a context per parallel branch. Before the fix, the first `destroy` trapped the instance every time; after it, runs that destroy hundreds of contexts complete normally across the portable (CPU/Metal) and CUDA drivers, including multi-hour GPU sweeps.
