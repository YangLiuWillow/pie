# `scripts/imagegen` — verification harness for the image/video generation effort

Reference implementations, golden dumps and a diff tool for the milestones in
[`.wiki/imagegen/design.md`](../../.wiki/imagegen/design.md) (D15 = verification,
§4 = milestone order).  **Nothing here touches the Rust tree.**  Everything is
Python that produces *reference numbers*; the pie side reproduces them.

Large artifacts live **outside the repo**, under
`$PIE_IMAGEGEN_GOLDEN` (default `/root/.cache/pie-imagegen/golden/`).

```
scripts/imagegen/
  mini_dit_ref.py     M0  synthetic mini-DiT reference (no real model, no download)
  mini_dit_parity.py  M0  drives pie's `mini-dit` row against that reference
  zimage_golden.py    M1  Z-Image-Turbo golden + miniature forward
  zimage_vae_parity.py M1 drives pie's `vae.decode` reading FROM A GUEST
  flux2_golden.py     M2  FLUX.2-klein-4B golden + miniature forward
  wan22_golden.py     M3  Wan 2.2 TI2V-5B golden + miniature forwards
  h3_golden.py        M5  MiniMax H3 miniature forward (vendored reference)
  h3_parity.py        M5  drives pie's `minimax-h3-mini` row against it
  vendor/minimax_h3/      a dependency-free transcription of H3's DiT --
                          sglang's own package cannot be imported here
  golden_common.py        shared tap/hook/manifest plumbing
  compare.py              npz-vs-npz diff with tolerance gates
```

---

## 1. Environment

`/root/.venv/imagegen` (uv-created, CPython 3.12.3).  System CUDA 13.0, driver
595.91.07, 4x RTX PRO 6000 Blackwell (sm_120, 97 GB each).

| package | version |
|---|---|
| torch | **2.14.0+cu130** (`torch.version.cuda == 13.0`) |
| torchvision | 0.29.0+cu130 |
| diffusers | **0.40.0** |
| transformers | 5.16.1 |
| tokenizers | 0.23.2 |
| accelerate | 1.14.0 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.30.0 |
| numpy | 2.5.2 |
| Pillow | 12.3.0 |
| imageio | 2.37.4 |
| imageio-ffmpeg | 0.6.0 |
| sentencepiece | 0.2.2 |
| protobuf | 7.36.1 |
| ftfy | 6.3.1 |
| einops | 0.8.2 |
| opencv-python-headless | 5.0.0.93 |

`torch.cuda.get_arch_list()` = `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`,
so sm_120 is covered natively.

Rebuild:

```bash
uv venv --python 3.12 ~/.venv/imagegen
uv pip install --python ~/.venv/imagegen/bin/python \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match torch torchvision
uv pip install --python ~/.venv/imagegen/bin/python \
  diffusers transformers accelerate safetensors "huggingface_hub[cli]" \
  Pillow numpy imageio imageio-ffmpeg sentencepiece protobuf ftfy einops \
  opencv-python-headless
```

**The cuBLAS symlink fix in `/root/SETUP_INSTRUCTIONS.md` was NOT needed.**  The
`cu130` wheel set ships `nvidia-cublas-cu13` that matches the system CUDA 13.0
runtime; a bf16 GEMM plus SDPA both ran clean on GPU 0 with no `LD_PRELOAD` and no
symlink surgery.  If a future wheel bump reintroduces the mismatch, apply the fix
to `~/.venv/imagegen/lib/python3.12/site-packages/nvidia/cu13/lib`.

**No vendoring was required** — diffusers 0.40.0 already ships `ZImagePipeline`,
`ZImageTransformer2DModel`, `Flux2KleinPipeline`, `Flux2Pipeline`,
`Flux2Transformer2DModel`, `AutoencoderKLFlux2`, `WanPipeline`,
`WanTransformer3DModel` and `AutoencoderKLWan`.  `scripts/imagegen/vendor/` does
not exist and is not needed at this diffusers pin.

---

## 2. Model downloads (HF cache `~/.cache/huggingface/hub`)

Downloaded in full (all files, `hf download <repo>`):

| repo | bytes on disk | note |
|---|---:|---|
| `Tongyi-MAI/Z-Image-Turbo` | 32,899,676,972 (32.9 GB) | fp32 transformer + Qwen3-4B encoder + FLUX 16-ch VAE |
| `black-forest-labs/FLUX.2-klein-4B` | 23,740,013,690 (23.7 GB) | `Flux2KleinPipeline`, Qwen3 encoder, `AutoencoderKLFlux2` |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 34,203,030,526 (34.2 GB) | `WanPipeline`, umT5-xxl, `AutoencoderKLWan` (4,16,16) z=48 |

Snapshot revisions: Z-Image `f332072a…`, FLUX.2-klein-4B `e7b7dc27…`,
Wan2.2-TI2V-5B-Diffusers `b8fff731…`.

Probed but **not** downloaded (all readable with the token in
`~/.cache/huggingface/token`; sizes from `files_metadata`):

| repo | gated | files | size |
|---|---|---:|---:|
| `Lightricks/LTX-2.5` | `auto` (accepted) | 17 | 200.9 GB |
| `MiniMaxAI/MiniMax-H3` | no | 280 | 498.5 GB |
| `tencent/HunyuanImage-3.0` | no | 84 | 168.7 GB |
| `black-forest-labs/FLUX.2-dev` | `auto` (accepted) | 39 | 177.6 GB |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | no | 49 | 126.2 GB |

Every one of these returned metadata successfully, i.e. **the token has read access
to all five** — the two `gated: auto` repos are already accepted for this account.
Total if all were pulled: ~1.17 TB against ~1.9 TB free (was 1.9 TB before the
91 GB above).  Diffusers-format siblings worth knowing about:
`Lightricks/LTX-2.5-Diffusers` exists; MiniMax and HunyuanImage ship custom code only.

---

## 3. `mini_dit_ref.py` — the M0 synthetic reference

No real model, no download, CPU-only, ~1 s.  It is the smallest graph that
exercises exactly the M0 substrate patterns and nothing else.

```bash
python mini_dit_ref.py            # --init --dump --euler
python mini_dit_ref.py --out-dir /somewhere/else
```

### Architecture (see `config.json` for the machine-readable copy)

hidden 256, 4 heads x head_dim 64, SwiGLU ratio 2 (hidden 512), RMSNorm QK-norm
(eps 1e-6), LayerNorm-no-affine (eps 1e-6), 3-axis interleaved RoPE
`dims [16,24,24]`, `theta 10000`.

| block | type | sequence it sees | modulation |
|---|---|---|---|
| 0 | single-stream joint | `[text(8) ‖ image(64)]` = 72 rows, one sequence | 6-param adaLN-Zero, `Linear(SiLU(temb))` |
| 1 | MM-DiT double-stream | text and image in **separate** rectangles, **joint** attention over the same 72-row order | 6-param **per stream**, separate `img_adaLN`/`txt_adaLN` |
| 2 | Wan-style cross-attn | image (64) only: self-attn → cross-attn → FFN | 6-param, `mod_table[6,256] + Linear(SiLU(temb)).view(B,6,256)` |

- Text stream ends after block 1 (block 2 and the head are image-only), so the
  head only ever emits velocity for patch rows.
- RoPE positions: text token `i` at `(i, 0, 0)`; image token at `(0, h, w)`.
  Applied to q and k **after** QK-norm.  Block 2's cross-attention gets **no** RoPE
  (the Wan contract) and its `norm_cross` is a LayerNorm **with** affine and **no**
  modulation; its cross residual is **ungated**.
- Cross-attn context: 16 rows of width 512, projected by one packed
  `Linear(512, 2*256)`.
- Patchify: `latent [B,16,16,16]` → `reshape(B,C,Hp,p,Wp,p)` →
  `permute(0,2,4,1,3,5)` → `[B, 64, 64]`, feature order `(c, ph, pw)`,
  token index `i = h*8 + w`.  Head is `LayerNorm-no-affine → x*(1+scale)+shift →
  Linear(256 → 64)` then the inverse permutation.
- Modulation chunk orders (load-bearing, easy to permute wrongly):
  blocks 0/1 `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`;
  block 2 `(shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn)`;
  final `(shift, scale)`.  Form is `x*(1+scale)+shift`, residual `x + gate*y`.
- Timestep embedding: `sinusoid(t, 256, max_period=10000)` = `[sin(all) ‖ cos(all)]`
  in fp32; each adaLN is `Linear(SiLU(temb))`.

Init is deterministic from seed 0 (parameters visited in **sorted name order**,
one `torch.Generator(0)`):
Linear weight `[out,in]` ~ `N(0, (1.0/sqrt(in))^2)`; bias ~ `N(0, 0.02^2)`;
QK-norm and LayerNorm gains = `1 + N(0, 0.02^2)`; `blocks.2.mod_table` ~ `N(0, 0.5^2)`.
This makes each block move the residual by **28–51 %** (measured), so a broken
block cannot pass a parity gate by accident.

### Tensor naming scheme (`mini_dit.safetensors`, 4,771,264 params, fp32)

```
x_embedder.{weight,bias}                          [256,64] [256]
blocks.0.adaLN.{weight,bias}                      [1536,256] [1536]
blocks.0.attn.qkv.{weight,bias}                   [768,256] [768]
blocks.0.attn.{norm_q,norm_k}                     [64]            # RMSNorm QK gains
blocks.0.attn.out.{weight,bias}                   [256,256] [256]
blocks.0.mlp.{gate_proj,up_proj}.{weight,bias}    [512,256] [512] # SwiGLU w1, w3
blocks.0.mlp.down_proj.{weight,bias}              [256,512] [256] # SwiGLU w2
blocks.1.{img,txt}_adaLN.{weight,bias}            [1536,256] [1536]
blocks.1.{img,txt}_attn.qkv.{weight,bias}         [768,256] [768]
blocks.1.{img,txt}_attn.{norm_q,norm_k}           [64]
blocks.1.{img,txt}_attn.out.{weight,bias}         [256,256] [256]
blocks.1.{img,txt}_mlp.{gate_proj,up_proj,down_proj}.{weight,bias}
blocks.2.mod_table                                [6,256]         # Wan scale_shift_table
blocks.2.adaLN.{weight,bias}                      [1536,256] [1536]
blocks.2.self_attn.qkv.{weight,bias}              [768,256] [768]
blocks.2.self_attn.{norm_q,norm_k}                [64]
blocks.2.self_attn.out.{weight,bias}              [256,256] [256]
blocks.2.norm_cross.{weight,bias}                 [256]           # LayerNorm WITH affine
blocks.2.cross_attn.q.{weight,bias}               [256,256] [256]
blocks.2.cross_attn.kv.{weight,bias}              [512,512] [512] # packed k|v from 512-wide ctx
blocks.2.cross_attn.{norm_q,norm_k}               [64]
blocks.2.cross_attn.out.{weight,bias}             [256,256] [256]
blocks.2.mlp.{gate_proj,up_proj,down_proj}.{weight,bias}
final_adaLN.{weight,bias}                         [512,256] [512]
final_proj.{weight,bias}                          [64,256] [64]
```

`config.json` carries the full `tensors` map, every chunk order, and the schedule.

### Dump keys — `mini_dit_dump_{fp32,bf16}.npz` (103 tensors each)

Fixed inputs: `latent [2,16,16,16]` and `text [2,8,256]` and `context [2,16,512]`
from `torch.Generator(1234)`, `timestep = [500., 250.]` (batch 2, different
timesteps).  Both files hold the same keys; `_bf16` is the bf16-emulated run stored
upcast to fp32.

```
in.latent [2,16,16,16]   in.text [2,8,256]   in.context [2,16,512]
in.timestep [2]          in.txt_pos [8,3]    in.img_pos [64,3]
rope.{cos,sin}_txt [8,32]    rope.{cos,sin}_img [64,32]
temb [2,256]   temb_silu [2,256]   patches [2,64,64]   x_embed [2,64,256]

b0.mod [2,1536]  b0.mod.{shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp} [2,256]
b0.in [2,72,256]        b0.norm1_out [2,72,256]
b0.{q_raw,k_raw,v} [2,4,72,64]      b0.{q_qknorm,k_qknorm} [2,4,72,64]
b0.{q_rope,k_rope} [2,4,72,64]      b0.attn_heads [2,4,72,64]
b0.attn_out [2,72,256]  b0.x_after_attn  b0.norm2_out  b0.mlp_out  b0.out [2,72,256]
b0.out_txt [2,8,256]    b0.out_img [2,64,256]

b1.{img,txt}_mod [2,1536]   b1.{img,txt}_mod.<same six names> [2,256]
b1.{img,txt}_norm1_out      b1.img_{q_rope,k_rope,v} [2,4,64,64]
b1.txt_{q_rope,k_rope,v} [2,4,8,64]     b1.joint_attn_heads [2,4,72,64]
b1.{img,txt}_attn_out       b1.{img,txt}_after_attn
b1.{img,txt}_norm2_out      b1.{img,txt}_mlp_out
b1.out_img [2,64,256]       b1.out_txt [2,8,256]

b2.mod_proj [2,6,256]   b2.mod [2,6,256]
b2.mod.{shift_msa,scale_msa,gate_msa,shift_ffn,scale_ffn,gate_ffn} [2,256]
b2.in  b2.norm1_out     b2.self_{q_rope,k_rope,v} [2,4,64,64]
b2.self_attn_heads [2,4,64,64]  b2.self_attn_out  b2.x_after_self
b2.cross_norm_out   b2.cross_q [2,4,64,64]  b2.cross_{k,v} [2,4,16,64]
b2.cross_attn_heads [2,4,64,64] b2.cross_attn_out  b2.x_after_cross
b2.norm3_out  b2.mlp_out  b2.out [2,64,256]

final.mod [2,512]  final.mod.shift [2,256]  final.mod.scale [2,256]
final.norm_out [2,64,256]   final.tokens [2,64,64]   velocity [2,16,16,16]
```

### Euler keys — `mini_dit_euler_{fp32,bf16}.npz` (427 tensors)

4 Euler flow-matching steps from `torch.Generator(7)` noise, sigmas
`[1.0, 0.75, 0.5, 0.25, 0.0]`, `t = sigma * 1000`, update `x += (s[i+1]-s[i])*v`:

```
euler.x_init [2,16,16,16]   euler.sigmas [5]
euler.t{0..3} [2]           euler.v{0..3} [2,16,16,16]
euler.x{1..4} [2,16,16,16]  euler.latent [2,16,16,16]
euler.s{0..3}.<every key from the dump list above>     # per-step intermediates
```

### bf16 emulation contract

`--dump`/`--euler` write both an exact-fp32 run and a bf16-emulated run.  bf16
rounds at every point a bf16 serving engine would: weights bf16; every Linear
input, weight, bias and output bf16; SwiGLU product bf16.  Reductions stay fp32
and are rounded back: LayerNorm/RMSNorm, the modulation `x*(1+s)+b`, the gated
residual add, RoPE (cos/sin fp32), the timestep sinusoid, and attention scores +
softmax (probabilities are cast to bf16 before `p·v`).

Measured fp32-vs-bf16 drift on the fixed inputs: worst max-abs 0.088, worst
relative 9.7e-3, worst cosine 0.99995 — a usable gate at
`--tol 0.1 --rel-tol 0.02 --cos-tol 0.9999`.

### Artifacts — `/root/.cache/pie-imagegen/golden/mini-dit/`

| file | bytes | md5 |
|---|---:|---|
| `config.json` | 5,995 | `9eeb9e43a9600eac66b233fcc61ac57d` |
| `mini_dit.safetensors` | 19,091,976 | `865dcbe3832b0f9fe31eb8544a738758` |
| `mini_dit_dump_fp32.npz` | 6,640,440 | `61e3f63d836bb820f8bbe0d2dcc37ddd` |
| `mini_dit_dump_bf16.npz` | 6,640,440 | `8ddf3e5b244c0bf9cad3d9266dd1152a` |
| `mini_dit_euler_fp32.npz` | 26,900,586 | `67e5976b6eadf30f78255de0725b6475` |
| `mini_dit_euler_bf16.npz` | 26,900,586 | `b53374ab535517f03d7b98eb25ac15db` |

### The pie side — `mini_dit_parity.py`

The other half of the M0 loop: it turns the golden's *inputs* into the case
JSON the `mini-dit-parity` inferlet takes, runs it, turns its JSON answer back
into an `.npz` under the golden's own key names, and diffs the two with
`compare.py` at the bf16 gate above.

```bash
# the artifact the row serves (the SKU name is `<text>-<weights>-kv-<kv>`)
cargo build -p pie --features cuda
pie model import "$PIE_IMAGEGEN_GOLDEN/mini-dit/" --sku mini-dit-bf16-kv-bf16 \
    --out ~/.cache/pie-imagegen/mini-dit.zt

# one step, then the four-step Euler schedule
python mini_dit_parity.py all --out /tmp/mini-dit-parity
python mini_dit_parity.py all --out /tmp/mini-dit-parity --euler
```

`case`, `collect` and `compare` are separate subcommands so a pie-side answer
produced any other way can be diffed too; `all` is the four in order.  One
batch element is one run (the golden's batch of 2 carries two timesteps), and
the harness stacks them back into the golden's `[2, ...]` shapes.  Patchify is
checked against the reference's own `patches` tensor on every invocation.

Both runs pass the gate (one step: cos 0.99996, max-abs 0.047; four Euler
steps: worst cos 0.99996, max-abs 0.049 on the velocities, 0.024 on the
latents).  The config the row is served under keeps `[engine] graphs = "off"`
so the parity walks eagerly.

**Bisecting a mismatch.**  `--tap <dump key>` on `run`, `collect` and
`compare` reads an INTERMEDIATE out in the velocity's place: `run` sets the
family's `PIE_MINI_DIT_TAP` knob (`crates/models/src/mini_dit/forward.rs`,
`Tap`) — the plan plants its readout seam on that rectangle, so the artifact
must be re-imported under the same environment — `collect` lays the pie rows
out the way the golden's tensor is shaped (a `[B, H, N, DH]` head tensor is
transposed back; a joint `[txt || img]` rectangle takes its caption rows from
the guest's caption-lane readout), and `compare` diffs that one key.

```bash
PIE_MINI_DIT_TAP=b0.norm1_out pie model import "$PIE_IMAGEGEN_GOLDEN/mini-dit/" \
    --sku mini-dit-bf16-kv-bf16 --out ~/.cache/pie-imagegen/mini-dit.zt --force
python mini_dit_parity.py run     --out /tmp/mini-dit-parity --tap b0.norm1_out --config ...
python mini_dit_parity.py collect --out /tmp/mini-dit-parity --tap b0.norm1_out
python mini_dit_parity.py compare --out /tmp/mini-dit-parity --tap b0.norm1_out
```

Every key the reference dumps between `x_embed` and `final.norm_out` is a
tap; `b0.in` and the `q_raw`/`v` heads included.  `run` also keeps the
server's log beside each answer (`pie_<b>.stderr`).

---

## 4. Model goldens

Each script has `--full` (real weights, GPU) and `--mini` (random-init miniature,
CPU).  With no flag it runs both.  Every run writes a `MANIFEST.json` next to the
outputs with sizes, md5s and the torch/diffusers/transformers versions used.

Common taps (in `golden_common.py`): `prepare_latents` → `noise.init*`;
`transformer.forward` call 0 → `dit.step0.in.*` and `dit.step0.out*`;
`scheduler.step` → `sched.x{1..N}`, with `latent.final` aliased to the last.

### `zimage_golden.py` → `/root/.cache/pie-imagegen/golden/z-image/`

Prompt `"a red bicycle leaning on a blue wall"`, seed 0 (CPU generator), 1024x1024,
**8 steps, guidance 0.0**, `max_sequence_length=512`, bf16.  Denoise took ~1.8 s.

Sigmas came out `[1.0, .95455, .900, .83333, .750, .64286, .500, .300, 0]` —
**exactly** the shift-3.0 values the study (`z-image.md` §K.2) predicts.
Prompt embeds are Qwen3 `hidden_states[-2]`, unpadded, `[16, 2560]` for this prompt.

Keys: `prompt_embeds.0 [16,2560]`, `prompt_embeds.lengths`, `noise.init [1,16,128,128]`,
`dit.step0.in.arg0.0 [16,1,128,128]` (latent, list-of-tensor calling convention),
`dit.step0.in.arg1 [1]` (t), `dit.step0.in.arg2.0 [16,2560]` (caption),
`dit.step0.out.0 [16,1,128,128]` (velocity), `sched.x{1..8}`, `latent.final`,
`sigmas [9]`, `timesteps [8]`, `image.rgb [1024,1024,3]`.

`--mini`: `ZImageTransformer2DModel(dim=256, n_layers=2, n_refiner_layers=2,
n_heads=4, axes_dims=[16,24,24], axes_lens=[256,64,64], cap_feat_dim=64)` —
2 noise-refiner + 2 context-refiner + 2 joint layers, 6,416,768 params.  Forward on
`x=[16,1,16,16]` (→ 64 image tokens at p=2), `cap=[8,64]`, `t=[500]`.

| file | bytes | md5 |
|---|---:|---|
| `zimage_golden.npz` | 25,498,702 | `efec906d7c786c18fcb70cc8a8707b97` |
| `zimage_golden.png` | 1,891,324 | `1acc842cdfbbfa719260561a878ed7f7` |
| `zimage_config.json` | 2,641 | `cad26e9472d8316797fb6b8eb054d991` |
| `zimage_mini.npz` | 35,856 | `d72d2751c83cafa3f266abf68049e0e0` |
| `zimage_mini.safetensors` | 25,677,496 | `fd71e61a6955381aff902cf10612c850` |
| `zimage_mini_config.json` | 7,278 | `3afce5520a90b635c009a890b866b64b` |

#### The pie side, from a guest — `zimage_vae_parity.py`

`zimage_golden.py --vae` dumps a 64x64x16 centre crop of `latent.final` beside
the pixels the reference VAE decodes it to
(`golden/z-image/zimage_vae/{latent,pixels,mean}.f32` + `shapes.json`).  The
engine gate `engine-cuda/tests/the_z_image_vae_answers_the_reference` feeds
that latent from the HOST; this script feeds it the way a guest can, through
`tests/inferlets/zimage-vae-parity`: the latent bound as the reading's
`Voxels` port CHANNEL (whose declared shape is the clip's box), the answer
read off `intrinsics::pixels()`, and the picture out through
`frames.from-channel` + `session.send-frames`.

```bash
cp ~/.pie/config.minidit.toml ~/.pie/config.zimage-vae.toml
# then edit: [model] model = the imported z-image-turbo artifact,
#            [engine] graphs = "on", max_model_len = 32768,
#            [server] port = something nothing else is using
python zimage_vae_parity.py all --out /tmp/zimage-vae \
    --config ~/.pie/config.zimage-vae.toml
```

Measured: **cos 0.999981, mean |err| 0.00225, max |err| 0.1526** against
`pixels.f32` — the same distance the host-fed gate reports (0.99998 / 0.0023),
so the guest road costs nothing.  `/tmp/zimage-vae/zimage-vae.png` is the
picture: the golden's red bicycle against its blue wall, 512x512.

`--no-pixels` takes the zero-copy road only (the PNG, no rows lifted); the two
roads produce a byte-identical PNG.

### `flux2_golden.py` → `/root/.cache/pie-imagegen/golden/flux2/`

`Flux2KleinPipeline` (klein-4B is `_class_name: Flux2KleinPipeline`, **not**
`Flux2Pipeline`), same prompt/seed, 1024x1024, **4 steps** (the distilled default).
klein-4B's `transformer/config.json` has `guidance_embeds: false`, `num_layers: 5`,
`num_single_layers: 20`, `num_attention_heads: 24`, `joint_attention_dim: 7680`
— the guidance scale is explicitly ignored by the pipeline.

Text embeddings are Qwen3 layers **(9, 18, 27)** concatenated → `[1, 512, 7680]`,
with a 4-axis `text_ids [1,512,4]` (text lives on axis 3, `l = arange(L)`).

Keys: `prompt_embeds [1,512,7680]`, `text_ids [1,512,4]`,
`noise.init.0 [1,4096,128]` (packed) and `noise.init.1 [1,4096,4]` (latent ids),
`dit.step0.in.{hidden_states,encoder_hidden_states,timestep,img_ids,txt_ids}`,
`dit.step0.out [1,4096,128]`, `sched.x{1..4}`, `latent.final`, `sigmas [5]`,
`timesteps [4]`, `image.rgb`.

`--mini`: `Flux2Transformer2DModel(num_layers=2, num_single_layers=2,
num_attention_heads=2, attention_head_dim=128, joint_attention_dim=192,
guidance_embeds=True)` — head_dim stays 128 because `sum(axes_dims_rope)=128`.
Runs 64 target tokens + **2 references of 64 tokens** at RoPE `T = 10` and `T = 20`
(`num_ref_tokens=128`), so the reference `T`-offsets are exercised.  6,604,288 params.

| file | bytes | md5 |
|---|---:|---|
| `flux2_golden.npz` | 60,969,856 | `128e9be25bf3a1aa073eb4b90929902b` |
| `flux2_golden.png` | 1,936,947 | `aef3b174e01c79b4444ed283e49a873a` |
| `flux2_config.json` | 2,344 | `3993447f0ab5cd868938f1fc84f26608` |
| `flux2_mini.npz` | 226,946 | `fa365df134ae5775ec3afb1c480e89aa` |
| `flux2_mini.safetensors` | 26,422,928 | `c8127a7a22a40e2c863e48a45ed80b62` |
| `flux2_mini_config.json` | 4,530 | `0a80cb07bd98017049c77edcd9e2f79e` |

#### `--vae` — the autoencoder alone

`AutoencoderKLFlux2` in fp32 (`force_upcast`) over the centre **32x32 tokens**
of `--full`'s `latent.final`.  Both sides cut at the **128-wide /16 grid** the
transformer itself holds, which is where `models::flux_2::vae` puts its port:
the VAE codes at 32 channels on a /8 grid and the pipeline packs a 2x2 block of
those into one 128-channel cell (`_patchify_latents`), normalised by the frozen
`BatchNorm2d(128)` (`(x - running_mean)/sqrt(running_var + eps)`).  There is no
`scaling_factor`/`shift_factor` on this family.

```text
decode: latent * bn_std + bn_mean -> _unpatchify_latents -> vae.decode
encode: vae.encode(...).mean      -> _patchify_latents   -> (x - bn_mean)/bn_std
```

`flux2_vae/{latent,pixels,mean}.f32` + `shapes.json` are the same planes as raw
little-endian f32 rows of channels in `(h, w)` order — what the Rust gate loads:

```bash
CUDA_VISIBLE_DEVICES=2 python flux2_golden.py --vae
# the gate reads the row's ARTIFACT, not the snapshot: a serving load may not
# apply the `Unary` the BatchNorm planes are stated through
pie model import <the FLUX.2-klein-4B snapshot> --sku flux2-klein-4b-bf16-kv-bf16 \
    --out ~/.cache/pie-imagegen/flux2-klein-4b.zt --force
CUDA_VISIBLE_DEVICES=2 cargo test -p engine-cuda --features cuda \
    --test the_flux_2_vae_answers_the_reference -- --nocapture
```

Measured (bf16 pie vs the fp32 golden): decode `[1,32,32] -> [1,512,512]` cos
0.999994, mean |err| 0.0017, max 0.040 (not one of 786 432 values past 0.05);
encode `[1,512,512] -> [1,32,32]` cos 0.999957, mean |err| 0.0069, max 0.115.

#### The pie side — `flux2_parity.py`

`tests/inferlets/flux2-parity` is the `flux2-mini` row's guest: one denoise
step over three lanes of one group — the text lane (`context` port, the raw
`[32, 192]` stack), the target lane (`latents`, 64 rows) and ONE reference
lane (`latents` again, both references' 128 rows, at their own `T`
offsets) — every lane binding `timestep` (`σ·1000`, i.e. 500) and
`guidance` (4.0, raw), the target lane alone reading out `velocity()`.
`flux2_parity.py` turns `flux2_mini.npz`'s inputs into the case, runs it,
and diffs pie's `[1, 64, 128]` against `mini.out.0[:, :64]` (the reference
discards the reference rows' predictions; pie never computes them).

```bash
# the artifact: the golden dir needs a `config.json` and a tokenizer
# beside the weights (the snapshot's `tokenizer/{tokenizer,tokenizer_config}.json`)
pie model import "$PIE_IMAGEGEN_GOLDEN/flux2/" --sku flux2-mini-bf16-kv-bf16 \
    --out ~/.cache/pie-imagegen/flux2-mini.zt
python flux2_parity.py all --out /tmp/flux2-parity --config ~/.pie/config.flux2-mini.toml
```

Measured (bf16 pie vs the fp32 golden): max-abs 0.0059, rel 0.0044, cos
0.99999 — under the mini-dit gate (`--tol 0.1 --rel-tol 0.02 --cos-tol 0.9999`).

#### The REAL row — `flux2_klein_parity.py`

`tests/inferlets/flux2-klein-parity` is the `flux2-klein-4b` row against the
same dump: the `text` reading over the family's chat template (the ids are
checked against `text.input_ids` exactly), one `denoise` step over the
golden's own step-0 inputs, one independent step per sigma from the
reference's own latent, and the four-step Euler trajectory, then both final
latents through the diffusers VAE.

```bash
# ~3 min to import, ~2 min to run
pie model import ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/*/ \
    --sku flux2-klein-4b-bf16-kv-bf16 --out ~/.cache/pie-imagegen/flux2-klein-4b.zt
CUDA_VISIBLE_DEVICES=0 python flux2_klein_parity.py all \
    --out /tmp/flux2-klein-parity --config ~/.pie/config.flux2-klein.toml
```

The config needs `[engine] max_model_len = 32768` (rows × submit depth),
`[model] model` at the artifact, and — the one that is not obvious —
`[runtime] submit_deadline = "10s"` with `silence_timeout = "300s"`: at the
50 ms default the cohort gate seals the denoise group's FIRST frame with
the image lane alone and the velocity comes back unconditioned (cos ≈ 0.535
against the golden, cos 0.9999 against a no-text reference). `run` refuses a
config that does not state it.

Measured (bf16 pie vs the bf16 golden, `graphs = "on"`):

| reading | cos | gate |
|---|---|---|
| `text.hidden` (20 rows × 3072) | 0.999993 | ≥ 0.999 ✓ |
| `dit.step0.out` | 0.999552 | ≥ 0.999 ✓ |
| `probe.step{0,1,2,3}.out` | 0.999552 / 0.999125 / 0.999615 / 0.999767 | ≥ 0.999 ✓ |
| `latent.final` (four Euler steps) | 0.997903 | reported |
| PSNR(`pie.png`, `golden.png`) | 34.87 dB | reported |

The trajectory is REPORTED, not gated: klein is distilled to four steps
whose last takes σ 0.767 → 0, which amplifies a velocity difference about
sixfold, and bf16 alone moves the step-0 velocity by 4e-4 in cosine. The
SAME diffusers transformer in fp32 reads cos 0.999618 on the velocity and
**0.997510** on `latent.final` against this bf16 dump — i.e. an exact fp32
computation is FARTHER from the golden than pie is. The per-step velocities
are what the model is gated on.

### `wan22_golden.py` → `/root/.cache/pie-imagegen/golden/wan22/`

`WanPipeline` on TI2V-5B, 480x832, **17 frames**, **8 steps**, seed 0, bf16 DiT with
the **VAE forced to fp32** (parity requires it, `wan22.md` §K.2).
`model_index.json` has `boundary_ratio: null` (single backbone) and
`expand_timesteps: true`.

Confirmed live: the transformer receives `timestep [1, 1950]` — a **per-token**
timestep, 1950 = 5 latent frames x 15 x 26 patch grid — which is the TI2V pattern
D6/D3 must support.  For pure T2V it is uniformly 999 at step 0 (zeros appear only
in the I2V/first-frame-conditioned case).  umT5 embeds are `[1,512,4096]` with
exactly 11 non-zero rows (truncate-then-**zero**-pad, unmasked).
Scheduler is `UniPCMultistepScheduler`; sigmas came out
`[0.999999, .97225, .93758, .89301, .83361, .75050, .62594, .41861, 0]`.

Keys: `prompt_embeds`, `negative_prompt_embeds`, `prompt_embeds.nonzero_rows`,
`noise.init [1,48,5,30,52]`, `dit.step0.in.{hidden_states,timestep,encoder_hidden_states}`,
`dit.step0.out`, `sched.x{1..8}`, `latent.final`, `sigmas [9]`, `timesteps [8]`,
`vae.decode.in [1,48,5,30,52]`, `vae.decode.out [1,3,17,480,832]`, `frames.u8_shape`.

`--mini`: two random-init `WanTransformer3DModel`s, each forwarded twice (scalar
timestep `[1]` and TI2V per-token timestep `[1,320]`):
- `nano` — `attention_head_dim=24` (rope split `d-4*(d//6)` → `[8,8,8]`), 93,584 params
- `d128` — `attention_head_dim=128` (the real `[44,42,42]` split), 2,227,008 params

Both on latent `[1,16,5,16,16]` → S = 320 tokens, context `[1,32,64]`.

| file | bytes | md5 |
|---|---:|---|
| `wan22_golden.npz` | 126,117,934 | `8ded636cff5fffa8c63d0010f7e70439` |
| `wan22_golden.mp4` | 129,186 | `ef55e6377f0426f41d679155d45008a5` |
| `wan22_frames.npy` | 20,367,488 | `6239bd8693fe9c714edf290fd6e647b7` |
| `wan22_frame0.png` | 652,441 | `fd7c3ee53c089709ec80f3f84d492761` |
| `wan22_config.json` | 4,390 | `c86f6f3a472b968de24fe1f0f9f809e6` |
| `wan22_mini.npz` | 513,878 | `4393141452e7ba1c4f4f18ae0879373b` |
| `wan22_mini_nano.safetensors` | 380,704 | `743e316070fbbf918e556b9e43d92b97` |
| `wan22_mini_d128.safetensors` | 8,914,648 | `5707cde610525dbcec5ca1c95f21ec16` |
| `wan22_mini_config.json` | 11,474 | `aab5a938c58f7864cbe2a03fe69bf47d` |

### `h3_golden.py` → `/root/.cache/pie-imagegen/golden/minimax_h3/`

**Miniature only, and the reference is VENDORED.** `sglang.multimodal_gen`
cannot be imported in this checkout (its `__init__` pulls in starlette and the
rest of the serving stack), so `vendor/minimax_h3/modeling.py` transcribes the
eager arithmetic of `runtime/models/dits/minimax_h3.py` from sglang commit
`6cee9285a3dd43e1f0270818aef7bf01a0568863`, class by class, with the upstream
line numbers in its header. Only eager branches are kept: no TP, no
quantization, no fused kernels, no adaLN cache, no sparse attention.

`--mini` builds the study's §D.3 configuration (2 blocks, 1 refiner, dim 128,
2 heads x 64, ffn 256, text_dim 64, 8 rope frequencies → 48 of 64 rotated;
863,048 params) over a 21-row packed sequence: 3 text rows, 8 video rows
(a 2 x 4 x 4 latent under the (1,2,2) patch), 6 audio rows (3 ticks, stereo,
channel-major) and 4 keyframe rows. The four unique timesteps are
`[0.35, 0.999, 0.62, 1.0]` — video, the pinned visual condition, audio, and the
ref2va audio-reference slot nothing in FL2VA claims.

It also **checks** the claim `crates/models/src/minimax_h3/forward.rs` makes
about the packed row order: pie's `[text | video | audio | reference]` (lanes by
stream code) and the reference's `[text | refs | audio | video]` answer at
cos 1.0, because the joint attention is unmasked and every row's rotary
coordinates travel with it.

The checkpoint it writes uses the OFFICIAL tensor spellings — `qkv_proj`
re-interleaved per head — so pie's import runs the same de-interleave the 66 GB
flagship needs.

Keys: `mini.in.{text_hidden,refined_text,video_rows,audio_rows,reference_rows,
positions,token_tags,inverse_indices,unique_timesteps}`,
`mini.out.{video,audio}`.

### The pie side — `h3_parity.py`

```bash
python h3_golden.py --mini
pie model import <dir with h3_mini.safetensors> --sku minimax-h3-mini-bf16-kv-bf16
python h3_parity.py all --out /tmp/h3-parity --config ~/.pie/config.h3-mini.toml
```

The guest (`tests/inferlets/h3-parity`) runs the `refine` pass (the text lane
alone) and then one `denoise` step of FOUR lanes in one attention group — text,
video, audio, reference — one pipeline each, one timestep cell per pass. Use a
PRIVATE config with its own `[server] port` and an `[engine] max_model_len` at
least the packed row count times the submit depth.

Measured (fp32 reference vs pie's bf16 trunk, one denoise step on one GPU):

| tensor | shape | max-abs | rel | cos |
|---|---|---:|---:|---:|
| `mini.out.refined_text` | (3, 128) | 0.00025 | 0.0039 | 0.99999254 |
| `mini.out.video` | (8, 96) | 0.00025 | 0.0027 | 0.99999653 |
| `mini.out.audio` | (6, 32) | 0.00020 | 0.0025 | 0.99999699 |

Gate: `--tol 0.1 --rel-tol 0.02 --cos-tol 0.9999`.

---

## 5. `compare.py`

```bash
python compare.py A.npz B.npz [--keys 'b0.*'] [--tol 1e-3] [--rel-tol 1e-4] \
                  [--cos-tol 0.9999] [--sort-by rel] [--quiet] [--allow-missing] \
                  [--floor 0] [--floor-frac 1e-3]
```

Per tensor: `max-abs`, `rel` = `||a-b|| / ||b||`, `max-rel-el` (worst element-wise
relative error over elements of B above `max(--floor, --floor-frac * max|B|)`,
default `max(0, 1e-3 * max|B|)`, so near-zero elements do not dominate it) and
cosine similarity.  Exit 1 on any tripped gate, on a shape mismatch, or on differing key
sets (unless `--allow-missing`).  Example:

```
$ python compare.py mini_dit_dump_bf16.npz mini_dit_dump_fp32.npz \
    --tol 0.1 --rel-tol 0.02 --cos-tol 0.9999 --quiet
compared 103 tensors  worst max-abs 0.08775  worst rel 0.009689  worst cos 0.999953099
PASS
```

For the Z-Image RMSNorm bit-exactness gate (`z-image.md` §K.2 asks for
`torch.equal` on norm outputs, not a tolerance) use `--tol 0 --keys '*norm*'`.
