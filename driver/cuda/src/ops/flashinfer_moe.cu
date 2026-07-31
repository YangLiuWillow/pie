#include "ops/flashinfer_moe.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>

#include "cutlass_fused_moe_kernels.cuh"

namespace pie_cuda_driver::ops {
namespace {

namespace ck = tensorrt_llm::kernels::cutlass_kernels;
namespace ce = tensorrt_llm::cutlass_extensions;
namespace tk = tensorrt_llm::kernels;

using Runner = ck::CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>;

bool env_truthy(const char* value) {
    if (value == nullptr || value[0] == '\0') return false;
    return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
           value[0] == 't' || value[0] == 'T' || value[0] == 'o' ||
           value[0] == 'O';
}

struct RunnerState {
    std::once_flag init_once;
    std::unique_ptr<Runner> runner;
    bool ready = false;
    std::exception_ptr init_error;
};

RunnerState& state() {
    constexpr int kMaxCudaDevices = 16;
    static std::array<RunnerState, kMaxCudaDevices> states;
    int device = 0;
    cudaError_t st = cudaGetDevice(&device);
    if (st != cudaSuccess) {
        throw std::runtime_error(
            std::string("flashinfer CUTLASS MoE: cudaGetDevice failed: ") +
            cudaGetErrorString(st));
    }
    if (device < 0 || device >= kMaxCudaDevices) {
        throw std::runtime_error(
            "flashinfer CUTLASS MoE: CUDA device index exceeds runner cache");
    }
    return states[device];
}

bool log_enabled() {
    return env_truthy(std::getenv("PIE_NEMOTRON_FLASHINFER_MOE_LOG"));
}

bool use_supported_tactic_selection() {
    const char* v = std::getenv("PIE_NEMOTRON_FLASHINFER_MOE_SELECT");
    return v != nullptr && std::strcmp(v, "supported") == 0;
}

bool use_raw_tactic_selection() {
    const char* v = std::getenv("PIE_NEMOTRON_FLASHINFER_MOE_SELECT");
    return v != nullptr && std::strcmp(v, "raw") == 0;
}

std::optional<ce::CutlassGemmConfig> select_first_profile(
    const std::vector<ce::CutlassGemmConfig>& configs,
    const char* name) {
    if (configs.empty()) return std::nullopt;
    if (log_enabled()) {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] FlashInfer MoE %s selected first profile "
            "total=%zu\n",
            name, configs.size());
    }
    return configs.front();
}

std::optional<ce::CutlassGemmConfig> first_supported(
    Runner& runner,
    const std::vector<ce::CutlassGemmConfig>& configs,
    std::optional<ce::CutlassGemmConfig::EpilogueFusionType> fusion,
    int supported_index,
    const char* name) {
    int seen = 0;
    int supported = 0;
    int index = -1;
    int selected_index = -1;
    std::optional<ce::CutlassGemmConfig> selected;
    for (const auto& cfg : configs) {
        ++index;
        if (fusion && cfg.epilogue_fusion_type != *fusion) continue;
        ++seen;
        if (runner.queryOccupancyForConfig(cfg) > 0) {
            if (supported == supported_index) {
                selected = cfg;
                selected_index = index;
            }
            ++supported;
        }
    }
    if (log_enabled()) {
        if (selected) {
            std::fprintf(
                stderr,
                "[pie-driver-cuda] FlashInfer MoE %s selected "
                "supported_index=%d raw_index=%d supported=%d seen=%d total=%zu\n",
                name, supported_index, selected_index, supported, seen, configs.size());
        } else {
            std::fprintf(
                stderr,
                "[pie-driver-cuda] FlashInfer MoE %s no tactic for "
                "supported_index=%d supported=%d seen=%d total=%zu\n",
                name, supported_index, supported, seen, configs.size());
        }
    }
    return selected;
}

std::optional<ce::CutlassGemmConfig> select_raw_profile(
    const std::vector<ce::CutlassGemmConfig>& configs,
    int raw_index,
    const char* name) {
    if (configs.empty()) return std::nullopt;
    const int index = std::min(
        std::max(0, raw_index),
        static_cast<int>(configs.size()) - 1);
    if (log_enabled()) {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] FlashInfer MoE %s selected raw_index=%d "
            "total=%zu\n",
            name, index, configs.size());
    }
    return configs[static_cast<std::size_t>(index)];
}

const char* fusion_name(ce::CutlassGemmConfig::EpilogueFusionType fusion) {
    switch (fusion) {
    case ce::CutlassGemmConfig::EpilogueFusionType::NONE: return "none";
    case ce::CutlassGemmConfig::EpilogueFusionType::FINALIZE: return "finalize";
    }
    return "unknown";
}

int env_index(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return 0;
    return std::max(0, std::atoi(v));
}

std::optional<ce::CutlassGemmConfig::EpilogueFusionType> requested_gemm2_fusion() {
    const char* v = std::getenv("PIE_NEMOTRON_FLASHINFER_MOE_GEMM2");
    if (v == nullptr || v[0] == '\0' || std::strcmp(v, "auto") == 0) {
        return std::nullopt;
    }
    if (std::strcmp(v, "none") == 0) {
        return ce::CutlassGemmConfig::EpilogueFusionType::NONE;
    }
    if (std::strcmp(v, "finalize") == 0) {
        return ce::CutlassGemmConfig::EpilogueFusionType::FINALIZE;
    }
    throw std::runtime_error(
        "PIE_NEMOTRON_FLASHINFER_MOE_GEMM2 must be auto, none, or finalize");
}

void log_config(const char* name, const ce::CutlassGemmConfig& cfg) {
    if (!log_enabled()) return;
    std::fprintf(
        stderr,
        "[pie-driver-cuda] FlashInfer MoE %s tactic: fusion=%s "
        "tma=%d swap_ab=%d sm=%d tile80=%d tile90=%d mainloop=%d "
        "epilogue=%d cluster=%d split_k=%d stages=%d\n",
        name,
        fusion_name(cfg.epilogue_fusion_type),
        cfg.is_tma_warp_specialized ? 1 : 0,
        cfg.swap_ab ? 1 : 0,
        cfg.sm_version,
        static_cast<int>(cfg.tile_config_sm80),
        static_cast<int>(cfg.tile_config_sm90),
        static_cast<int>(cfg.mainloop_schedule),
        static_cast<int>(cfg.epilogue_schedule),
        static_cast<int>(cfg.cluster_shape),
        cfg.split_k_factor,
        cfg.stages);
}

Runner& get_runner() {
    RunnerState& s = state();
    std::call_once(s.init_once, [&] {
        try {
            auto runner = std::make_unique<Runner>();
            auto gemm1 = runner->getTactics(ck::MoeGemmId::GEMM_1);
            auto gemm2 = runner->getTactics(ck::MoeGemmId::GEMM_2);
            std::optional<ce::CutlassGemmConfig> best_gemm1;
            std::optional<ce::CutlassGemmConfig> best_gemm2;
            if (use_raw_tactic_selection()) {
                best_gemm1 = select_raw_profile(
                    gemm1,
                    env_index("PIE_NEMOTRON_FLASHINFER_MOE_GEMM1_INDEX"),
                    "GEMM1");
                best_gemm2 = select_raw_profile(
                    gemm2,
                    env_index("PIE_NEMOTRON_FLASHINFER_MOE_GEMM2_INDEX"),
                    "GEMM2");
            } else if (use_supported_tactic_selection()) {
                best_gemm1 = first_supported(
                    *runner, gemm1, std::nullopt,
                    env_index("PIE_NEMOTRON_FLASHINFER_MOE_GEMM1_INDEX"),
                    "GEMM1");
                best_gemm2 = first_supported(
                    *runner, gemm2, requested_gemm2_fusion(),
                    env_index("PIE_NEMOTRON_FLASHINFER_MOE_GEMM2_INDEX"),
                    "GEMM2");
            } else {
                best_gemm1 = select_first_profile(gemm1, "GEMM1");
                best_gemm2 = select_first_profile(gemm2, "GEMM2");
            }
            if (!best_gemm1 || !best_gemm2) {
                throw std::runtime_error(
                    "flashinfer CUTLASS MoE: no supported BF16 tactics");
            }
            log_config("GEMM1", *best_gemm1);
            log_config("GEMM2", *best_gemm2);
            runner->setTactic(best_gemm1, best_gemm2);
            s.runner = std::move(runner);
            s.ready = true;
        } catch (...) {
            s.init_error = std::current_exception();
        }
    });

    if (!s.ready) {
        if (s.init_error) std::rethrow_exception(s.init_error);
        throw std::runtime_error("flashinfer CUTLASS MoE: runner not initialized");
    }
    return *s.runner;
}

ck::MOEParallelismConfig parallelism_config(int tp_size, int tp_rank) {
    return ck::MOEParallelismConfig(std::max(1, tp_size), tp_rank, 1, 0);
}

}  // namespace

bool flashinfer_cutlass_moe_enabled() {
    static const bool enabled =
        env_truthy(std::getenv("PIE_NEMOTRON_FLASHINFER_MOE"));
    return enabled;
}

std::size_t flashinfer_cutlass_moe_workspace_bytes(
    int num_rows,
    int hidden_size,
    int inter_size,
    int num_experts,
    int experts_per_token,
    int tp_size,
    int tp_rank) {
    if (num_rows <= 0 || hidden_size <= 0 || inter_size <= 0 ||
        num_experts <= 0 || experts_per_token <= 0) {
        return 0;
    }
    Runner& runner = get_runner();
    return runner.getWorkspaceSize(
        num_rows,
        hidden_size,
        inter_size,
        num_experts,
        experts_per_token,
        ck::ActivationType::Relu2,
        parallelism_config(tp_size, tp_rank),
        false,
        false,
        false,
        false,
        false);
}

bool flashinfer_cutlass_moe_bf16_relu2(
    const std::uint16_t* input,
    const std::int32_t* token_selected_experts,
    const float* token_final_scales,
    const std::uint16_t* fc1_expert_weights,
    const std::uint16_t* fc2_expert_weights,
    std::uint16_t* output,
    std::uint8_t* workspace,
    std::size_t workspace_bytes,
    std::int32_t* unpermuted_row_to_permuted_row,
    int num_rows,
    int hidden_size,
    int inter_size,
    int num_experts,
    int experts_per_token,
    int tp_size,
    int tp_rank,
    cudaStream_t stream) {
    if (!flashinfer_cutlass_moe_enabled()) return false;
    if (input == nullptr || token_selected_experts == nullptr ||
        token_final_scales == nullptr || fc1_expert_weights == nullptr ||
        fc2_expert_weights == nullptr || output == nullptr ||
        workspace == nullptr || unpermuted_row_to_permuted_row == nullptr) {
        return false;
    }
    const std::size_t needed = flashinfer_cutlass_moe_workspace_bytes(
        num_rows, hidden_size, inter_size, num_experts, experts_per_token,
        tp_size, tp_rank);
    if (needed == 0 || workspace_bytes < needed) return false;

    Runner& runner = get_runner();
    ck::QuantParams quant_params{};
    tk::LoraParams lora_params{};
    ck::MoeMinLatencyParams min_latency_params{};
    runner.runMoe(
        input,
        nullptr,
        false,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        nullptr,
        ck::ActivationParams(ck::ActivationType::Relu2),
        fc2_expert_weights,
        nullptr,
        quant_params,
        num_rows,
        hidden_size,
        hidden_size,
        inter_size,
        num_experts,
        experts_per_token,
        reinterpret_cast<char*>(workspace),
        output,
        unpermuted_row_to_permuted_row,
        parallelism_config(tp_size, tp_rank),
        false,
        false,
        lora_params,
        false,
        false,
        false,
        min_latency_params,
        false,
        stream);
    return true;
}

// Probe helper for the doctor entry point below. Takes the activation as a plain
// int so the diagnostic code needs none of this file's internal type aliases.
// ActivationType: Gelu=0 Relu=1 Silu=2 Swiglu=3 Geglu=4 SwigluBias=5 Relu2=6.
std::size_t moe_probe_workspace_bytes(
    int act_raw, int num_rows, int hidden, int inter, int experts, int topk) {
    Runner& runner = get_runner();
    // THE THIRD PARAMETER IS fc1_output_size, NOT inter_size. They differ for
    // gated activations: moe_gemm_template_dispatch.h:830 states
    //   fc1_out_size = is_gated_activation ? inter_size * 2 : inter_size
    // and runMoe() takes inter_size in the corresponding slot, so the two calls
    // want DIFFERENT values. Passing inter_size to both looks right, works for
    // Relu2, and makes every gated shape fail with "Could not find valid config"
    // -- which reads like a kernel limitation and is not one.
    const auto act = static_cast<ck::ActivationType>(act_raw);
    const int64_t fc1_output_size = ck::isGatedActivation(act) ? 2LL * inter : inter;
    return runner.getWorkspaceSize(
        num_rows, hidden, fc1_output_size, experts, topk,
        act,
        parallelism_config(/*tp_size=*/1, /*tp_rank=*/0),
        false, false, false, false, false);
}

// Dispatch-level probe: actually RUN the lower-level variable-M grouped GEMM
// (MoeGemmRunner::moeGemm, EpilogueOpDefault = no activation) at a given
// (n, k, experts) shape, for every config the runner offers that is NOT
// TMA warp-specialized (those launchers are not compiled — see the banner in
// the workspace probe below). getWorkspaceSize lies at dispatch time; this
// does not. Returns a one-line summary; never throws.
//
// This is the probe for the non-gated CUTLASS route to the prefill MoE:
// up-projection as a PLAIN n=2*inter GEMM, elementwise SwiGLU between,
// down-projection n=hidden — sidestepping the gated-epilogue minimum that
// requires the uncompiled TMA-WS kernels.
std::string moe_dispatch_probe_run(int64_t n, int64_t k, int experts, int64_t rows) {
    using GemmRunner =
        ck::MoeGemmRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>;
    std::string line;
    void* A = nullptr; void* B = nullptr; void* C = nullptr;
    int64_t* offsets_d = nullptr;
    auto cleanup = [&] {
        if (A) cudaFree(A);
        if (B) cudaFree(B);
        if (C) cudaFree(C);
        if (offsets_d) cudaFree(offsets_d);
        cudaGetLastError();  // clear any sticky error for later probes
    };
    try {
        GemmRunner runner;
        if (cudaMalloc(&A, rows * k * 2) != cudaSuccess ||
            cudaMalloc(&B, static_cast<int64_t>(experts) * k * n * 2) != cudaSuccess ||
            cudaMalloc(&C, rows * n * 2) != cudaSuccess ||
            cudaMalloc(&offsets_d, experts * sizeof(int64_t)) != cudaSuccess) {
            cleanup();
            return "alloc failed (GPU busy?)";
        }
        cudaMemset(A, 0, rows * k * 2);
        cudaMemset(B, 0, static_cast<int64_t>(experts) * k * n * 2);
        // Rows spread evenly: cumulative inclusive per expert.
        std::vector<int64_t> offsets_h(experts);
        for (int e = 0; e < experts; ++e) {
            offsets_h[e] = rows * (e + 1) / experts;
        }
        cudaMemcpy(offsets_d, offsets_h.data(), experts * sizeof(int64_t),
                   cudaMemcpyHostToDevice);

        const auto configs = runner.getConfigs(false);
        int tried = 0, ran = 0;
        std::string first_ok, first_err;
        for (const auto& cfg : configs) {
            if (runner.isTmaWarpSpecialized(cfg)) continue;
            ++tried;
            ck::GroupedGemmInput<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                 __nv_bfloat16> in;
            in.A = static_cast<__nv_bfloat16 const*>(A);
            in.B = static_cast<__nv_bfloat16 const*>(B);
            in.C = static_cast<__nv_bfloat16*>(C);
            in.total_tokens_including_expert = offsets_d;
            in.num_rows = rows;
            in.n = n;
            in.k = k;
            in.num_experts = experts;
            in.stream = nullptr;
            in.gemm_config = cfg;
            try {
                runner.moeGemm(in, {});
                cudaError_t sync = cudaDeviceSynchronize();
                if (sync == cudaSuccess) {
                    ++ran;
                    if (first_ok.empty()) first_ok = cfg.toString();
                } else if (first_err.empty()) {
                    first_err = cudaGetErrorString(sync);
                    cudaGetLastError();
                }
            } catch (const std::exception& e) {
                if (first_err.empty()) {
                    first_err = e.what();
                    const auto nl = first_err.find('\n');
                    if (nl != std::string::npos) first_err = first_err.substr(0, nl);
                }
                cudaGetLastError();
            }
        }
        cleanup();
        line = std::to_string(ran) + "/" + std::to_string(tried) +
               " non-TMA configs RAN";
        if (ran > 0) {
            std::string c = first_ok;
            if (c.size() > 70) c = c.substr(0, 70) + "...";
            line += " (first: " + c + ")";
        } else if (!first_err.empty()) {
            if (first_err.size() > 90) first_err = first_err.substr(0, 90) + "...";
            line += " — first error: " + first_err;
        }
        return line;
    } catch (const std::exception& e) {
        cleanup();
        std::string msg(e.what());
        const auto nl = msg.find('\n');
        if (nl != std::string::npos) msg = msg.substr(0, nl);
        return std::string("probe setup failed: ") + msg;
    }
}

}  // namespace pie_cuda_driver::ops

// -----------------------------------------------------------------------------
// Diagnostic probe, reachable from `pie driver cuda-native doctor`.
//
// WHY THIS EXISTS. Reading `case ActivationType::Swiglu:` in the activation
// dispatch switch is NOT evidence that the runner can serve a given MoE shape:
// the workspace calculation separately requires that at least one TMA
// warp-specialized GEMM config be valid for (hidden, inter, experts, gated?),
// and if none is, getWorkspaceSize throws "Could not find valid config".
// That is exactly how the 2026-07-29 Qwen3-MoE attempt failed, after a code
// reading said it should work.
//
// This answers the question in under a second with NO MODEL LOAD, so the
// activation-vs-shape distinction can be settled before writing integration
// code. Fills `out` with a human-readable report; never throws.
extern "C" void pie_driver_cuda_moe_probe(char* out, int out_len) {
    if (out == nullptr || out_len <= 0) return;
    std::string r;
    // ActivationType enum values (common.h): Swiglu=3, Relu2=6.
    constexpr int kSwiglu = 3;
    constexpr int kRelu2 = 6;

    struct Shape { const char* name; int hidden; int inter; int experts; int topk; };
    // Qwen3-Coder-30B-A3B is the first row. The rest vary ONE dimension at a
    // time so a failure can be attributed rather than guessed at.
    const Shape shapes[] = {
        {"qwen3-30b-a3b   ", 2048, 768,  128, 8},
        {"  inter 768->1536", 2048, 1536, 128, 8},
        {"  inter 768->2048", 2048, 2048, 128, 8},
        {"  experts 128->64", 2048, 768,  64,  8},
        {"  hidden 2048->4096", 4096, 768, 128, 8},
        // Repeat of row 1. If this SUCCEEDS where row 1 failed, the runner
        // carries order-dependent state (getMaxWorkspaceSize caches on
        // num_experts_) and a "no valid config" on a cold first call says
        // nothing about whether the shape is supported.
        {"qwen3-30b-a3b AGAIN", 2048, 768,  128, 8},
    };
    const struct { const char* name; int act; } acts[] = {
        {"swiglu(gated)", kSwiglu},
        {"relu2 (plain)", kRelu2},
    };

    r += "  moe-probe (CutlassMoeFCRunner<bf16,bf16>, num_rows=512, tp=1)\n";
    // "OK" HERE MEANS THE SHAPE IS DESCRIBABLE, NOT THAT IT IS RUNNABLE.
    // This calls getWorkspaceSize, which only needs a config DESCRIPTOR to exist.
    // On 2026-07-29 a gated shape reported "OK, workspace 40 MiB" and then ABORTED
    // THE DRIVER at dispatch with "Please recompile with support for hopper by
    // passing 90-real as an arch to build_wheel.py" -- because the Hopper TMA
    // warp-specialized launchers are NOT compiled into this build. The vendored
    // tree ships only the template (moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl)
    // and no .cu instantiating it; defining COMPILE_HOPPER_TMA_GROUPED_GEMMS
    // without generating those TUs fails at link with hundreds of undefined
    // tma_warp_specialized_generic_moe_gemm_kernelLauncher<Sm90,...> symbols.
#ifdef COMPILE_HOPPER_TMA_GROUPED_GEMMS
    r += "    [hopper TMA-WS launchers: COMPILED — 'OK' below is trustworthy]\n";
#else
    r += "    [hopper TMA-WS launchers: NOT COMPILED — 'OK' below means the shape\n"
         "     is describable, NOT that a kernel exists. A gated dispatch will\n"
         "     abort the driver. Do not read 'OK' as support.]\n";
#endif
    for (const auto& s : shapes) {
        for (const auto& a : acts) {
            r += "    ";
            r += s.name;
            r += "  ";
            r += a.name;
            r += " : ";
            try {
                const std::size_t bytes =
                    pie_cuda_driver::ops::moe_probe_workspace_bytes(
                        a.act, 512, s.hidden, s.inter, s.experts, s.topk);
                if (bytes == 0) {
                    r += "0 bytes (runner declined without error)";
                } else {
                    r += "OK, workspace ";
                    r += std::to_string(bytes >> 20);
                    r += " MiB";
                }
            } catch (const std::exception& e) {
                std::string msg(e.what());
                // Keep only the first line; TllmException carries a stack trace.
                const auto nl = msg.find('\n');
                if (nl != std::string::npos) msg = msg.substr(0, nl);
                if (msg.size() > 110) msg = msg.substr(0, 110) + "...";
                r += "FAIL: " + msg;
            }
            r += "\n";
        }
    }

    // DISPATCH-LEVEL probe: run the lower-level variable-M grouped GEMM
    // (no activation epilogue) at the shapes the non-gated prefill route
    // needs. Unlike the section above, a line here saying configs RAN is
    // ground truth — the kernel executed and synchronized.
    r += "  moe-dispatch-probe (MoeGemmRunner<bf16> moeGemm, rows=512, "
         "non-TMA configs only)\n";
    struct DShape { const char* name; int64_t n; int64_t k; int experts; };
    const DShape dshapes[] = {
        // The two GEMMs of the non-gated Qwen3-Coder-30B-A3B prefill route.
        {"up   n=1536 k=2048 E=128 (gate_up as plain 2I)", 1536, 2048, 128},
        {"down n=2048 k=768  E=128", 2048, 768, 128},
        // One control at a shape the workspace probe calls OK for Relu2.
        {"ctl  n=768  k=2048 E=128 (inter as-is)", 768, 2048, 128},
    };
    for (const auto& d : dshapes) {
        r += "    ";
        r += d.name;
        r += " : ";
        r += pie_cuda_driver::ops::moe_dispatch_probe_run(d.n, d.k, d.experts, 512);
        r += "\n";
    }
    std::snprintf(out, static_cast<std::size_t>(out_len), "%s", r.c_str());
}
