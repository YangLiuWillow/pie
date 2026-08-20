#include "response_subpass.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "device_buffer.hpp"
#include "kernels/dist.hpp"
#include "kernels/entropy.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/logprobs.hpp"
#include "model/qwen3_forward.hpp"
#include "sampler_type.hpp"

namespace pie_cuda_driver {

namespace {

constexpr std::uint32_t TYPE_DIST     = static_cast<std::uint32_t>(SamplerType::Dist);
constexpr std::uint32_t TYPE_RAWLOG   = static_cast<std::uint32_t>(SamplerType::RawLogits);
constexpr std::uint32_t TYPE_LOGPROB  = static_cast<std::uint32_t>(SamplerType::Logprob);
constexpr std::uint32_t TYPE_LOGPROBS = static_cast<std::uint32_t>(SamplerType::Logprobs);
constexpr std::uint32_t TYPE_ENTROPY  = static_cast<std::uint32_t>(SamplerType::Entropy);

}  // namespace

void gather_raw_logits(
    const ResponseSubpassContext& ctx,
    std::vector<pie_driver::PerRequestOutput>& per_req)
{
    const int V = ctx.vocab_size;
    const auto* h_qo   = ctx.qo_indptr.data();
    const auto* h_sptr = ctx.sampling_indptr.data();
    const auto* h_sidx = ctx.sampling_indices.data();

    // Pass 1: collect (req, source_row) for every RawLogits slot, in
    // slot-iteration order. `req_for_slot[i]` = the response request
    // index where the i-th gathered row's payload belongs.
    std::vector<std::int32_t> rows;
    std::vector<int> req_for_slot;
    rows.reserve(ctx.num_sampling);
    req_for_slot.reserve(ctx.num_sampling);
    for (int r = 0; r < ctx.R; ++r) {
        const std::uint32_t lo = h_sptr[r];
        const std::uint32_t hi = h_sptr[r + 1];
        const std::uint32_t qo_lo = h_qo[r];
        for (std::uint32_t k = lo; k < hi; ++k) {
            if (ctx.per_slot_type[k] != TYPE_RAWLOG) continue;
            rows.push_back(static_cast<std::int32_t>(qo_lo + h_sidx[k]));
            req_for_slot.push_back(r);
        }
    }
    if (rows.empty()) return;

    // Pass 2: one kernel launch + one D2H. Replaces the previous
    // per-slot `cudaMemcpy` loop, which serialized on the default
    // stream and incurred a launch overhead per slot.
    const std::size_t n = rows.size();
    auto d_rows   = DeviceBuffer<std::int32_t>::from_host(
        std::span<const std::int32_t>(rows));
    auto d_packed = DeviceBuffer<std::uint16_t>::alloc(n * V);
    kernels::launch_gather_bf16_rows(
        static_cast<const std::uint16_t*>(ctx.ws.logits.data()),
        d_rows.data(), d_packed.data(),
        static_cast<int>(n), V, /*stream=*/nullptr);
    const std::vector<std::uint16_t> h_packed = d_packed.to_host();

    // Pass 3: per-slot bf16 → f32 widening on host. Cheap (host-only,
    // V ≈ 150K → ~600 KB per slot) and amortizes one host alloc per
    // slot. Convention: place bf16 bits in the high 16 bits of the
    // f32 — matches `_process_raw_logits` in pie_driver.
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint16_t* src = h_packed.data() + i * V;
        std::vector<std::uint8_t> payload(V * sizeof(float));
        auto* out = reinterpret_cast<std::uint32_t*>(payload.data());
        for (int j = 0; j < V; ++j) {
            out[j] = static_cast<std::uint32_t>(src[j]) << 16;
        }
        per_req[req_for_slot[i]].logits.push_back(std::move(payload));
    }
}

void compute_entropy_slots(
    const ResponseSubpassContext& ctx,
    std::vector<pie_driver::PerRequestOutput>& per_req)
{
    const auto* h_qo   = ctx.qo_indptr.data();
    const auto* h_sptr = ctx.sampling_indptr.data();
    const auto* h_sidx = ctx.sampling_indices.data();

    std::vector<std::int32_t> ent_rows;
    std::vector<int> ent_req_idx;
    ent_rows.reserve(ctx.num_sampling);
    ent_req_idx.reserve(ctx.num_sampling);
    for (int r = 0; r < ctx.R; ++r) {
        const std::uint32_t qo_lo = h_qo[r];
        for (std::uint32_t k = h_sptr[r]; k < h_sptr[r + 1]; ++k) {
            if (ctx.per_slot_type[k] != TYPE_ENTROPY) continue;
            ent_rows.push_back(static_cast<std::int32_t>(qo_lo + h_sidx[k]));
            ent_req_idx.push_back(r);
        }
    }
    if (ent_rows.empty()) return;

    auto d_ent_rows = DeviceBuffer<std::int32_t>::from_host(
        std::span<const std::int32_t>(ent_rows));
    auto d_ent_out  = DeviceBuffer<float>::alloc(ent_rows.size());
    kernels::launch_entropy_bf16(
        ctx.ws.logits.data(), d_ent_rows.data(), d_ent_out.data(),
        static_cast<int>(ent_rows.size()),
        ctx.vocab_size, /*stream=*/nullptr);
    const auto h_ent = d_ent_out.to_host();
    for (std::size_t i = 0; i < ent_req_idx.size(); ++i) {
        per_req[ent_req_idx[i]].entropies.push_back(h_ent[i]);
    }
}

void compute_logprob_slots(
    const ResponseSubpassContext& ctx,
    const pie_driver::PieForwardRequestView& view,
    std::vector<pie_driver::PerRequestOutput>& per_req)
{
    const auto label_ids_view    = view.sampler_label_ids.as<std::uint32_t>();
    const auto label_indptr_view = view.sampler_label_indptr.as<std::uint32_t>();

    const auto* h_qo   = ctx.qo_indptr.data();
    const auto* h_sptr = ctx.sampling_indptr.data();
    const auto* h_sidx = ctx.sampling_indices.data();
    const auto* h_rns  = ctx.request_num_samplers.data();

    std::vector<std::int32_t> lp_rows;
    std::vector<std::int32_t> lp_label_indptr = {0};
    std::vector<std::int32_t> lp_label_ids;
    std::vector<int> lp_req_idx;

    std::uint32_t s_off = 0;
    for (int r = 0; r < ctx.R; ++r) {
        const std::uint32_t ns =
            (ctx.request_num_samplers.size() > static_cast<std::size_t>(r)) ? h_rns[r] : 0u;
        const std::uint32_t qo_lo = h_qo[r];
        const std::uint32_t lo = h_sptr[r];
        const std::uint32_t hi = h_sptr[r + 1];
        for (std::uint32_t k = lo; k < hi; ++k) {
            const std::uint32_t type = ctx.per_slot_type[k];
            if (type != TYPE_LOGPROB && type != TYPE_LOGPROBS) continue;
            const std::uint32_t s_idx = s_off + (k - lo);
            // sampler_label_indptr is CSR with length num_samplers+1.
            const std::uint32_t li_lo =
                (s_idx < label_indptr_view.size()) ? label_indptr_view[s_idx] : 0u;
            const std::uint32_t li_hi =
                (s_idx + 1 < label_indptr_view.size()) ? label_indptr_view[s_idx + 1] : li_lo;
            const int n_labels = static_cast<int>(li_hi) - static_cast<int>(li_lo);
            lp_rows.push_back(static_cast<std::int32_t>(qo_lo + h_sidx[k]));
            for (int t = 0; t < n_labels; ++t) {
                lp_label_ids.push_back(
                    static_cast<std::int32_t>(label_ids_view[li_lo + t]));
            }
            lp_label_indptr.push_back(
                static_cast<std::int32_t>(lp_label_ids.size()));
            lp_req_idx.push_back(r);
        }
        s_off += ns;
    }
    if (lp_rows.empty()) return;

    auto d_lp_rows    = DeviceBuffer<std::int32_t>::from_host(
        std::span<const std::int32_t>(lp_rows));
    auto d_lp_lindptr = DeviceBuffer<std::int32_t>::from_host(
        std::span<const std::int32_t>(lp_label_indptr));
    // Always allocate at least 1 element so the kernel gets a valid
    // pointer for the "labels for every slot are empty" edge case.
    auto d_lp_lids = lp_label_ids.empty()
        ? DeviceBuffer<std::int32_t>::alloc(1)
        : DeviceBuffer<std::int32_t>::from_host(
              std::span<const std::int32_t>(lp_label_ids));
    auto d_lp_out = DeviceBuffer<float>::alloc(
        std::max<std::size_t>(lp_label_ids.size(), 1));

    kernels::launch_logprobs_bf16(
        ctx.ws.logits.data(), d_lp_rows.data(), d_lp_lindptr.data(),
        d_lp_lids.data(), d_lp_out.data(),
        static_cast<int>(lp_rows.size()),
        ctx.vocab_size, /*stream=*/nullptr);

    std::vector<float> h_lp(lp_label_ids.size());
    if (!lp_label_ids.empty()) {
        CUDA_CHECK(cudaMemcpy(h_lp.data(), d_lp_out.data(),
                              sizeof(float) * h_lp.size(),
                              cudaMemcpyDeviceToHost));
    }

    for (std::size_t i = 0; i < lp_req_idx.size(); ++i) {
        const std::int32_t lo = lp_label_indptr[i];
        const std::int32_t hi = lp_label_indptr[i + 1];
        per_req[lp_req_idx[i]].logprobs.emplace_back(
            h_lp.data() + lo, h_lp.data() + hi);
    }
}

void compute_dist_slots(
    const ResponseSubpassContext& ctx,
    std::vector<pie_driver::PerRequestOutput>& per_req)
{
    const int V = ctx.vocab_size;
    const auto* h_qo   = ctx.qo_indptr.data();
    const auto* h_sptr = ctx.sampling_indptr.data();
    const auto* h_sidx = ctx.sampling_indices.data();

    // Sentinel: `temperature == 0` means "top-K raw logits" — the values
    // returned are the model's pre-softmax logits, not probabilities (SDK
    // `TopLogits` probe). Softmax slots and raw slots take different device
    // paths, so collect all TYPE_DIST slots in slot order first and merge
    // results back in that order — a request may mix both kinds.
    struct DistSlot {
        std::int32_t row;
        float temp;
        std::int32_t topk;
        int req;
    };
    std::vector<DistSlot> slots;
    slots.reserve(ctx.num_sampling);

    for (int r = 0; r < ctx.R; ++r) {
        const std::uint32_t lo = h_sptr[r];
        const std::uint32_t hi = h_sptr[r + 1];
        const std::uint32_t qo_lo = h_qo[r];
        for (std::uint32_t k = lo; k < hi; ++k) {
            if (ctx.per_slot_type[k] != TYPE_DIST) continue;
            // per_slot_top_k was already mapped (0 → V) upstream; clamp
            // once more for safety.
            const std::int32_t Tk =
                (ctx.per_slot_top_k[k] <= 0) ? V : ctx.per_slot_top_k[k];
            slots.push_back({static_cast<std::int32_t>(qo_lo + h_sidx[k]),
                             ctx.per_slot_temp[k], Tk, r});
        }
    }
    if (slots.empty()) return;

    std::vector<std::size_t> soft_idx;
    std::vector<std::size_t> raw_idx;
    for (std::size_t i = 0; i < slots.size(); ++i) {
        (slots[i].temp == 0.0f ? raw_idx : soft_idx).push_back(i);
    }

    // Per-slot result rows, indexed like `slots`; merged at the end so
    // `per_req[..].dists` keeps slot order.
    std::vector<std::vector<float>> values(slots.size());

    if (!soft_idx.empty()) {
        std::vector<std::int32_t> rows;
        std::vector<float> temps;
        rows.reserve(soft_idx.size());
        temps.reserve(soft_idx.size());
        for (const auto i : soft_idx) {
            rows.push_back(slots[i].row);
            temps.push_back(slots[i].temp);
        }
        auto d_rows  = DeviceBuffer<std::int32_t>::from_host(
            std::span<const std::int32_t>(rows));
        auto d_temps = DeviceBuffer<float>::from_host(
            std::span<const float>(temps));
        auto d_probs = DeviceBuffer<float>::alloc(
            soft_idx.size() * static_cast<std::size_t>(V));

        kernels::launch_softmax_temp_bf16(
            ctx.ws.logits.data(), d_rows.data(), d_temps.data(),
            d_probs.data(), static_cast<int>(soft_idx.size()), V,
            /*stream=*/nullptr);

        std::vector<float> h_probs = d_probs.to_host();
        for (std::size_t j = 0; j < soft_idx.size(); ++j) {
            values[soft_idx[j]].assign(h_probs.begin() + j * V,
                                       h_probs.begin() + (j + 1) * V);
        }
    }

    if (!raw_idx.empty()) {
        std::vector<std::int32_t> rows;
        rows.reserve(raw_idx.size());
        for (const auto i : raw_idx) rows.push_back(slots[i].row);
        auto d_rows   = DeviceBuffer<std::int32_t>::from_host(
            std::span<const std::int32_t>(rows));
        auto d_packed = DeviceBuffer<std::uint16_t>::alloc(
            raw_idx.size() * static_cast<std::size_t>(V));
        kernels::launch_gather_bf16_rows(
            static_cast<const std::uint16_t*>(ctx.ws.logits.data()),
            d_rows.data(), d_packed.data(),
            static_cast<int>(raw_idx.size()), V, /*stream=*/nullptr);
        const std::vector<std::uint16_t> h_packed = d_packed.to_host();

        // bf16 → f32 widening: bf16 bits in the high 16 bits of the f32
        // (same convention as the RawLogits sub-pass).
        for (std::size_t j = 0; j < raw_idx.size(); ++j) {
            const std::uint16_t* src = h_packed.data() + j * V;
            auto& dst = values[raw_idx[j]];
            dst.resize(V);
            auto* bits = reinterpret_cast<std::uint32_t*>(dst.data());
            for (int v = 0; v < V; ++v) {
                bits[v] = static_cast<std::uint32_t>(src[v]) << 16;
            }
        }
    }

    std::vector<std::pair<float, std::uint32_t>> scratch(V);
    for (std::size_t i = 0; i < slots.size(); ++i) {
        const auto& row = values[i];
        for (int j = 0; j < V; ++j) {
            scratch[j] = {row[j], static_cast<std::uint32_t>(j)};
        }
        const int K = slots[i].topk < V ? slots[i].topk : V;
        // Partial sort: top-K by value descending; tie-break by lower
        // id (matches torch.topk's stable behavior).
        std::partial_sort(
            scratch.begin(), scratch.begin() + K, scratch.end(),
            [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });
        std::vector<std::uint32_t> ids(K);
        std::vector<float> vals(K);
        for (int kk = 0; kk < K; ++kk) {
            ids[kk]  = scratch[kk].second;
            vals[kk] = scratch[kk].first;
        }
        per_req[slots[i].req].dists.emplace_back(std::move(ids), std::move(vals));
    }
}

}  // namespace pie_cuda_driver
