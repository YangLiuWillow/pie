#include "kernels.hpp"

#include <cmath>
#include <string>
#include <vector>

#include "../../device_tuning.hpp"
#include "../qwen3_5/decode_dispatch_mb.hpp"

namespace pie::metal::llama {

bool build_llama_psos(RawMetalContext& ctx, const std::string& kernels_dir,
                      const LlamaGeometry& g, LlamaPsos& out, std::string* err) {
    const std::string dir =
        kernels_dir.empty() || kernels_dir.back() == '/' ? kernels_dir : kernels_dir + "/";
    struct Spec {
        const char* file;
        std::string fn;
        Pso* dst;
    };
    // bf16 throughout: the activation dtype every ported M=1 kernel already uses.
    // The head width is the geometry's, not a literal: see `LlamaPsos::sdpa`.
    // A width with no instantiation fails here, by name, instead of running a
    // pipeline built for a different one.
    const std::string d = "_d_" + std::to_string(g.head_dim);
    const std::string q = g.quant.kernel_suffix();
    const std::string sdpa_name = "sdpa_vector_decode_bfloat16" + d;
    const std::string paged_name =
        "sdpa_paged_decode_bfloat16" + d + (g.kv_page_size == 32 ? "_p32" : "");
    const std::string tiled_name = "sdpa_paged_tiled_bfloat16" + d;
    std::vector<Spec> specs = {
        {"sdpa_vector.metal", sdpa_name, &out.sdpa},
        {"sdpa_paged.metal", paged_name, &out.sdpa_paged},
        {"sdpa_paged.metal", tiled_name, &out.sdpa_paged_tiled},
        {"row_gather.metal", "row_gather_bfloat16", &out.row_gather},
    };
    if (g.head_dim == 64 && g.kv_page_size == 32) {
        specs.push_back(
            {"sdpa_paged.metal", paged_name + "_sg8", &out.sdpa_paged_sg8});
    }
    // The head-sharing decode, when this checkpoint's geometry is one the
    // kernel is instantiated for. Asked through the same predicate `pso_for`
    // and `launch_shape` use, so a geometry can never be compiled-for and then
    // not selected, or selected and not compiled.
    if (sdpa_head_share_this_fire(g.head_dim, g.kv_page_size, g.n_q_heads,
                                  g.n_kv_heads, g.paged_kv_enabled)) {
        specs.push_back({"sdpa_paged.metal",
                         paged_name + "_h" + std::to_string(kSdpaHeadShare),
                         &out.sdpa_paged_hshare});
    }
    if (g.rope_freq_table) {
        specs.push_back({"rope.metal", "rope_neox_freqs_decode_bfloat16", &out.rope_freqs});
        specs.push_back({"rope.metal", "rope_neox_freqs_mb_bfloat16", &out.rope_freqs_mb});
    }
    // Only for a routed checkpoint. A dense one never dispatches these, and
    // compiling them anyway would let an unrelated shader error fail a load
    // that would otherwise have worked.
    if (g.is_moe()) {
        // The router at its own width, when the checkpoint gave it one. Same
        // entrypoint as the dense matvec the router otherwise shares -- only
        // the suffix differs, because only the bytes do.
        if (g.has_alt_router_quant()) {
            specs.push_back({"quantized_qmv.metal",
                             "affine_qmv_fast" + g.router_quant.kernel_suffix(),
                             &out.router_alt});
        }
        specs.push_back({"moe_route.metal", "router_topk_bfloat16", &out.router_topk});
        specs.push_back({"quantized_qmv.metal", "affine_qmv_routed" + q, &out.qmv_routed});
        specs.push_back({"moe_route.metal", "moe_route_sort", &out.moe_sort});
        specs.push_back({"moe_route.metal", "moe_route_gather", &out.moe_gather});
        specs.push_back({"moe_route.metal", "moe_combine_sorted", &out.moe_combine});
        // The batched form's three column tiles, at each of the two tile
        // widths `moe_tile_rows` can pick. `bm` is what the sort padded every
        // expert's run to -- naming it here would be a second statement of the
        // same number, so it is spelled from the shared table.
        for (int t = 0; t < 3; ++t) {
            const std::string routed_bm =
                "affine_qmm_t_routed" + q + "_bm_" +
                std::to_string(shared_kernels::kMoeTileWidths[t]);
            for (int i = 0; i < 3; ++i) {
                specs.push_back({"quantized_qmm_t.metal",
                                 routed_bm + "_bn_" + std::to_string(16 << i),
                                 &out.qmm_routed[t][i]});
            }
        }
    }
    for (const Spec& spec : specs) {
        std::string compile_error;
        *spec.dst = ctx.compile_pso_from_file(dir + spec.file, spec.fn.c_str(), &compile_error);
        if (!spec.dst->valid()) {
            if (err != nullptr) {
                *err = "llama PSO '" + spec.fn + "' (" + spec.file +
                       "): " + compile_error;
            }
            return false;
        }
    }
    // Not in the list above because it is conditional, and it is conditional on
    // exactly what `llama_sdpa_mma_this_fire` asks -- minus the row count,
    // which is a per-fire question and this is load time. Gating the COMPILE on
    // `sdpa_mma()` as well as the width is what makes `PIE_METAL_SDPA_MMA=0` a
    // complete way back: it removes the pipeline and every dispatch that would
    // have chosen it, so the switch cannot half-apply.
    //
    // Where it IS asked for, a failure is fatal rather than a fallback. The
    // matrix shape is 128 threads and the scalar one is 1024, and the fire's
    // choice between them is made in `launch_shape`, which is not handed a
    // `LlamaPsos` and so cannot notice an invalid one. Falling back silently
    // here would leave the grid describing a kernel other than the one that
    // runs -- wrong numbers, not a crash.
    if (sdpa_mma() && sdpa_mma_head_dim_supported(g.head_dim, /*with_sink=*/false)) {
        // `_p32` swaps two runtime integer divisions per staged element for a
        // shift and a mask -- 7.52 -> 6.88 ms/layer at 184 rows / 7424 ctx.
        // The test is exact equality, matching `paged_name` above: page size
        // is an unvalidated operator setting, so anything inferential here
        // would read the wrong slot rather than fail.
        const std::string mma_name =
            "sdpa_paged_mma_bfloat16" + d + (g.kv_page_size == 32 ? "_p32" : "");
        std::string compile_error;
        out.sdpa_paged_mma = ctx.compile_pso_from_file(
            dir + "sdpa_paged_mma.metal", mma_name.c_str(), &compile_error);
        if (!out.sdpa_paged_mma.valid()) {
            if (err != nullptr) {
                *err = "llama PSO '" + mma_name + "' (sdpa_paged_mma.metal): " +
                       compile_error;
            }
            return false;
        }
    }
    return true;
}

std::vector<float> llama3_inv_freq(const LlamaGeometry& g) {
    const int dims = g.rotary_dims();
    const int half = dims / 2;
    std::vector<float> inv_freq(std::size_t(half < 1 ? 1 : half), 0.0f);
    if (half < 1) return inv_freq;

    const float base = g.rope_theta;
    const float factor = g.rope_scaling_factor > 0.0f ? g.rope_scaling_factor : 1.0f;
    const float lo = g.rope_low_freq_factor;
    const float hi = g.rope_high_freq_factor;
    const float orig = float(g.rope_original_max_position);
    const float low_wavelen = orig / lo;
    const float high_wavelen = orig / hi;

    for (int i = 0; i < half; ++i) {
        // mlx's `_freqs` is base^(2i/dims) -- a WAVELENGTH-like quantity, the
        // reciprocal of the usual inv_freq. The schedule is expressed on it,
        // so it is computed on it and inverted once at the end.
        const float freq = std::pow(base, float(2 * i) / float(dims));
        const float wavelen = 2.0f * float(M_PI) * freq;
        float scaled = freq;
        if (wavelen > low_wavelen) {
            // Turns too slowly to extrapolate: interpolate by the whole factor.
            scaled = freq * factor;
        } else if (wavelen > high_wavelen) {
            // The ramp. Below `high_wavelen` the dimension is left alone, which
            // is the untouched `scaled = freq` this branch falls past.
            const float smooth = (orig / wavelen - lo) / (hi - lo);
            scaled = freq / ((1.0f - smooth) / factor + smooth);
        }
        inv_freq[std::size_t(i)] = scaled != 0.0f ? 1.0f / scaled : 0.0f;
    }
    return inv_freq;
}

}  // namespace pie::metal::llama
