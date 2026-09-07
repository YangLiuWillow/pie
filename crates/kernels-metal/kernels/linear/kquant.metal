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

// Sum over one Q5_K super-block: Q4_K plus a fifth-bit plane read per sub-block
// pair (bit 2p for the low nibble, 2p+1 for the high); the fifth bit adds 16
// before the affine minimum. Mirrors `decode_gguf_q5_k_block_into`.
template <typename T>
inline float kdot_q5k(const device uint8_t* blk, const device T* xg) {
  const float d = gguf_f16(blk);
  const float dmin = gguf_f16(blk + 2);
  const device uint8_t* plane = blk + 16;
  const device uint8_t* qs = blk + 48;
  float acc = 0.0f;
  for (int pair = 0; pair < 4; ++pair) {
    int sc_lo, m_lo, sc_hi, m_hi;
    q4k_scale_min(pair * 2, blk + 4, sc_lo, m_lo);
    q4k_scale_min(pair * 2 + 1, blk + 4, sc_hi, m_hi);
    const float d_lo = d * float(sc_lo);
    const float min_lo = dmin * float(m_lo);
    const float d_hi = d * float(sc_hi);
    const float min_hi = dmin * float(m_hi);
    const device uint8_t* packed = qs + pair * 32;
    const uint8_t bit_lo = uint8_t(1u << (2 * pair));
    const uint8_t bit_hi = uint8_t(1u << (2 * pair + 1));
    const int o = pair * 64;
    float plo = 0.0f, xlo = 0.0f, phi = 0.0f, xhi = 0.0f;
    for (int i = 0; i < 32; ++i) {
      const float xl = float(xg[o + i]);
      const float xh = float(xg[o + 32 + i]);
      const uint fifth_lo = (plane[i] & bit_lo) != 0 ? 16u : 0u;
      const uint fifth_hi = (plane[i] & bit_hi) != 0 ? 16u : 0u;
      plo = fma(float((packed[i] & 0x0F) + fifth_lo), xl, plo);
      xlo += xl;
      phi = fma(float((packed[i] >> 4) + fifth_hi), xh, phi);
      xhi += xh;
    }
    acc = fma(d_lo, plo, acc);
    acc = fma(-min_lo, xlo, acc);
    acc = fma(d_hi, phi, acc);
    acc = fma(-min_hi, xhi, acc);
  }
  return acc;
}

// Sum over one Q2_K super-block: sixteen sub-blocks with a 4-bit scale and a
// 4-bit min each, over one f16 d/dmin that CLOSE the block (bytes 80/82). The
// 64-byte payload is two 32-byte windows, each read four times at shifts
// 0/2/4/6. Mirrors `decode_gguf_q2_k_block_into`.
template <typename T>
inline float kdot_q2k(const device uint8_t* blk, const device T* xg) {
  const device uint8_t* scales = blk;
  const device uint8_t* qs = blk + 16;
  const float d = gguf_f16(blk + 80);
  const float dmin = gguf_f16(blk + 82);
  float acc = 0.0f;
  int out = 0;
  int sub = 0;
  for (int window = 0; window < 2; ++window) {
    const device uint8_t* q = qs + window * 32;
    for (int step = 0; step < 4; ++step) {
      const int shift = 2 * step;
      for (int hh = 0; hh < 2; ++hh) {
        const uint8_t packed = scales[sub++];
        const float dl = d * float(packed & 0x0F);
        const float ml = dmin * float(packed >> 4);
        float part = 0.0f, xsum = 0.0f;
        for (int l = 0; l < 16; ++l) {
          const float xv = float(xg[out++]);
          part = fma(float((q[hh * 16 + l] >> shift) & 3), xv, part);
          xsum += xv;
        }
        acc = fma(dl, part, acc);
        acc = fma(-ml, xsum, acc);
      }
    }
  }
  return acc;
}

// Q3_K's sixteen 6-bit scales, spliced from 12 bytes (biased by 32, the caller
// subtracts). Mirrors `gguf_q3_k_scales`.
inline void q3k_scales(const device uint8_t* raw, thread int* scales) {
  const uint a = (uint)raw[0] | ((uint)raw[1] << 8) | ((uint)raw[2] << 16) | ((uint)raw[3] << 24);
  const uint b = (uint)raw[4] | ((uint)raw[5] << 8) | ((uint)raw[6] << 16) | ((uint)raw[7] << 24);
  const uint top = (uint)raw[8] | ((uint)raw[9] << 8) | ((uint)raw[10] << 16) | ((uint)raw[11] << 24);
  const uint LOW = 0x0f0f0f0fu;
  const uint PAIRS = 0x03030303u;
  uint aux[4];
  aux[0] = (a & LOW) | ((top & PAIRS) << 4);
  aux[1] = (b & LOW) | (((top >> 2) & PAIRS) << 4);
  aux[2] = ((a >> 4) & LOW) | (((top >> 4) & PAIRS) << 4);
  aux[3] = ((b >> 4) & LOW) | (((top >> 6) & PAIRS) << 4);
  for (int i = 0; i < 16; ++i) {
    const uint byte = (aux[i / 4] >> (8 * (i % 4))) & 0xffu;
    scales[i] = int(as_type<char>((uint8_t)byte));
  }
}

// Sum over one Q3_K super-block: symmetric, sixteen sub-blocks, each element's
// third bit in a separate mask read one bit per (window, step) pair (inverted:
// a set bit means no borrow). Mirrors `decode_gguf_q3_k_block_into`.
template <typename T>
inline float kdot_q3k(const device uint8_t* blk, const device T* xg) {
  const device uint8_t* hmask = blk;
  const device uint8_t* qs = blk + 32;
  const float d = gguf_f16(blk + 108);
  int scales[16];
  q3k_scales(blk + 96, scales);
  float acc = 0.0f;
  int out = 0;
  int sub = 0;
  uint8_t selector = 1;
  for (int window = 0; window < 2; ++window) {
    const device uint8_t* q = qs + window * 32;
    for (int step = 0; step < 4; ++step) {
      const int shift = 2 * step;
      for (int hh = 0; hh < 2; ++hh) {
        const float dl = d * float(scales[sub++] - 32);
        float part = 0.0f;
        for (int l = 0; l < 16; ++l) {
          const int at = hh * 16 + l;
          const int borrow = (hmask[at] & selector) == 0 ? 4 : 0;
          const int code = int((q[at] >> shift) & 3) - borrow;
          part = fma(float(code), float(xg[out++]), part);
        }
        acc = fma(dl, part, acc);
      }
      selector <<= 1;
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

KQUANT_DEFINE(q2k, 84, kdot_q2k)
KQUANT_DEFINE(q3k, 110, kdot_q3k)
KQUANT_DEFINE(q4k, 144, kdot_q4k)
KQUANT_DEFINE(q5k, 176, kdot_q5k)
KQUANT_DEFINE(q6k, 210, kdot_q6k)

#define KQUANT_INSTANTIATE(TAG, NAME, ITYPE, RPW)                              \
  template [[host_name("kquant_matmul_" #TAG "_" #NAME "_r_" #RPW)]]           \
  [[kernel]] void kquant_matmul_##TAG<ITYPE, RPW>(                            \
      const device ITYPE*, const device uint8_t*, device ITYPE*,              \
      const constant int&, const constant int&, uint3, uint, uint);

KQUANT_INSTANTIATE(q2k, bfloat16, bfloat, 4)
KQUANT_INSTANTIATE(q3k, bfloat16, bfloat, 4)
KQUANT_INSTANTIATE(q4k, bfloat16, bfloat, 4)
KQUANT_INSTANTIATE(q5k, bfloat16, bfloat, 4)
KQUANT_INSTANTIATE(q6k, bfloat16, bfloat, 4)
KQUANT_INSTANTIATE(q2k, float16, half, 4)
KQUANT_INSTANTIATE(q3k, float16, half, 4)
KQUANT_INSTANTIATE(q4k, float16, half, 4)
KQUANT_INSTANTIATE(q5k, float16, half, 4)
KQUANT_INSTANTIATE(q6k, float16, half, 4)
