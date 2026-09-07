// GGUF K-quant decode-in-dot: the Metal twin of
// `kernels_cuda/kernels/linear/kquant.cuh`. A K-quant weight reaches here as
// one byte plane (`engine_metal::Run::maybe_stored` re-badges the stored Dense
// handle to `U8`); the scales live inside each 256-element super-block and are
// decoded in the dot, so the weight serves at the size it shipped.
//
// The per-scheme decode arithmetic is transcribed bit-for-bit from the host
// reference `checkpoint::executor::walk::decode_gguf_q{4,6}_k_block_into` — the
// oracle the kernel is diffed against. PR2 lands q4_k and q6_k (the Q4_K_M mix:
// q4_k bodies, q6_k `output.weight`); q2_k/q3_k/q5_k follow.

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

// Elements in one K-quant super-block, all five schemes.
constant constexpr int KSUPER = 256;

// The dispatch launches KNUM_SIMD simdgroups per threadgroup (KNUM_SIMD*32
// threads); each simdgroup folds ROWS_PER_WARP weight rows, and its 32 lanes
// stride that row's super-blocks. Kept equal to the Rust launch's `group.y`.
constant constexpr int KNUM_SIMD = 4;

// A little-endian f16 read from unaligned bytes (blocks are 144/210 wide, so
// the super-scales are not naturally aligned).
inline float gguf_f16(const device uint8_t* at) {
  ushort bits = (ushort)at[0] | ((ushort)at[1] << 8);
  return float(as_type<half>(bits));
}

// ggml `get_scale_min_k4`: the 6-bit sub-block scale+min for Q4_K/Q5_K,
// spliced from the shared 12 bytes. Sub-blocks 4-7 take their high bits from
// the bits sub-blocks 0-3 leave unused.
inline void q4k_scale_min(int sub, const device uint8_t* s, thread int& scale, thread int& mn) {
  if (sub < 4) {
    scale = s[sub] & 63;
    mn = s[sub + 4] & 63;
  } else {
    scale = (s[sub + 4] & 0x0F) | ((s[sub - 4] >> 6) << 4);
    mn = (s[sub + 4] >> 4) | ((s[sub] >> 6) << 4);
  }
}

// Sum over one Q4_K super-block of decode(w)_e * x_e, via the affine identity
// sum((d*sc*q - dmin*m) * x) = d*sc*sum(q*x) - dmin*m*sum(x), accumulated per
// sub-block. Payload byte i of pair p carries element 64p+i (low nibble) and
// 64p+32+i (high nibble); sub-block b uses pair b/2, nibble b&1.
template <typename T>
inline float kdot_q4k(const device uint8_t* blk, const device T* xg) {
  const float d = gguf_f16(blk);
  const float dmin = gguf_f16(blk + 2);
  const device uint8_t* qs = blk + 16;
  float acc = 0.0f;
  for (int b = 0; b < 8; ++b) {
    const int pair = b >> 1;
    const bool high = (b & 1) != 0;
    float part = 0.0f;
    float xsum = 0.0f;
    for (int i = 0; i < 32; ++i) {
      const float xv = float(xg[b * 32 + i]);
      xsum += xv;
      const uint8_t byte = qs[pair * 32 + i];
      const float q = high ? float(byte >> 4) : float(byte & 0x0F);
      part = fma(q, xv, part);
    }
    int scale, mn;
    q4k_scale_min(b, blk + 4, scale, mn);
    acc = fma(d * float(scale), part, acc);
    acc = fma(-(dmin * float(mn)), xsum, acc);
  }
  return acc;
}

// Sum over one Q6_K super-block of decode(w)_e * x_e. Symmetric (no min): the
// element is d*scale*(6-bit - 32). Two 128-element halves, four strided
// quarters each, two 16-element sub-blocks per quarter; the low nibble comes
// from `ql`, the top two bits from `qh`, the scale is a signed i8.
template <typename T>
inline float kdot_q6k(const device uint8_t* blk, const device T* xg) {
  const float d = gguf_f16(blk + 208);
  float acc = 0.0f;
  for (int hlf = 0; hlf < 2; ++hlf) {
    const device uint8_t* ql = blk + hlf * 64;
    const device uint8_t* qh = blk + 128 + hlf * 32;
    const device uint8_t* sc = blk + 192 + hlf * 8;
    for (int quarter = 0; quarter < 4; ++quarter) {
      for (int sub = 0; sub < 2; ++sub) {
        float part = 0.0f;
        for (int t = 0; t < 16; ++t) {
          const int i = sub * 16 + t;
          const float xv = float(xg[hlf * 128 + quarter * 32 + i]);
          const uint8_t byte = ql[i + 32 * (quarter & 1)];
          const uint low = (quarter < 2) ? (uint)(byte & 0x0F) : (uint)(byte >> 4);
          const uint top = ((uint)qh[i] >> (2 * quarter)) & 3u;
          const float q = float(int(low | (top << 4)) - 32);
          part = fma(q, xv, part);
        }
        const char cs = as_type<char>(sc[sub + 2 * quarter]);
        acc = fma(d * float(cs), part, acc);
      }
    }
  }
  return acc;
}

// One decode-in-dot GEMV kernel per scheme: `x` is [tokens, k], `w` is [n,
// row_bytes] of stored super-blocks, `y` is [tokens, n]. Grid: threadgroup
// (token, column-tile); KNUM_SIMD simdgroups each fold ROWS_PER_WARP rows, and
// the 32 lanes stride the row's super-blocks, folded by `simd_sum`.
#define KQUANT_DEFINE(TAG, BYTES, KDOT)                                        \
  template <typename T, int ROWS_PER_WARP>                                     \
  [[kernel]] void kquant_matmul_##TAG(                                         \
      const device T* x [[buffer(0)]],                                         \
      const device uint8_t* w [[buffer(1)]],                                   \
      device T* y [[buffer(2)]],                                               \
      const constant int& n [[buffer(3)]],                                     \
      const constant int& k [[buffer(4)]],                                     \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]) {                           \
    const int token = int(tid.x);                                             \
    const int blocks = k / KSUPER;                                            \
    const int row_bytes = blocks * (BYTES);                                   \
    const int out_row =                                                       \
        int(tid.y) * (KNUM_SIMD * ROWS_PER_WARP) + int(simd_gid) * ROWS_PER_WARP; \
    const device T* xrow = x + (ulong)token * (ulong)k;                       \
    float acc[ROWS_PER_WARP];                                                 \
    for (int r = 0; r < ROWS_PER_WARP; ++r) {                                 \
      acc[r] = 0.0f;                                                          \
    }                                                                         \
    for (int r = 0; r < ROWS_PER_WARP; ++r) {                                 \
      const int row = out_row + r;                                            \
      if (row >= n) {                                                         \
        continue;                                                            \
      }                                                                       \
      const device uint8_t* wrow = w + (ulong)row * (ulong)row_bytes;         \
      float a = 0.0f;                                                         \
      for (int g = int(simd_lid); g < blocks; g += 32) {                      \
        a += KDOT<T>(wrow + (ulong)g * (BYTES), xrow + (ulong)g * KSUPER);    \
      }                                                                       \
      acc[r] = a;                                                            \
    }                                                                         \
    for (int r = 0; r < ROWS_PER_WARP; ++r) {                                 \
      const float s = simd_sum(acc[r]);                                       \
      const int row = out_row + r;                                            \
      if (simd_lid == 0 && row < n) {                                         \
        y[(ulong)token * (ulong)n + row] = T(s);                              \
      }                                                                       \
    }                                                                         \
  }

KQUANT_DEFINE(q4k, 144, kdot_q4k)
KQUANT_DEFINE(q6k, 210, kdot_q6k)

#define KQUANT_INSTANTIATE(TAG, NAME, ITYPE, RPW)                              \
  template [[host_name("kquant_matmul_" #TAG "_" #NAME "_r_" #RPW)]]           \
  [[kernel]] void kquant_matmul_##TAG<ITYPE, RPW>(                            \
      const device ITYPE*, const device uint8_t*, device ITYPE*,              \
      const constant int&, const constant int&, uint3, uint, uint);

KQUANT_INSTANTIATE(q4k, bfloat16, bfloat, 4)
KQUANT_INSTANTIATE(q6k, bfloat16, bfloat, 4)
KQUANT_INSTANTIATE(q4k, float16, half, 4)
KQUANT_INSTANTIATE(q6k, float16, half, 4)
