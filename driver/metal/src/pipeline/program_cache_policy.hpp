#pragma once

// Which cached program to evict — the POLICY, separated from the cache so it
// can be tested without a Metal device.
//
// The decision is worth isolating because getting it wrong is silent in two
// different directions. Evict something still in use and a running fire loses
// its executable; refuse to evict when something WAS available and the cache
// stays full, which is the failure this whole change exists to remove.

#include <cstdint>
#include <optional>
#include <vector>

namespace pie::metal::pipeline {

/// One cached program, as the policy sees it.
struct ProgramCacheEntry {
    std::uint64_t hash = 0;
    /// `shared_ptr::use_count()` for the cached executable. **1 means the cache
    /// is the only owner**, so dropping it frees the entry now. Anything
    /// greater is held by a live instance or an in-flight fire, and that
    /// reference count IS the pin: those owners keep their own `shared_ptr`, so
    /// an evicted-but-running program finishes normally and only the next
    /// lookup misses.
    long use_count = 1;
    /// Monotonic tick of the last hit. Larger is more recent.
    std::uint64_t used_at = 0;
};

/// The least-recently-used entry the cache may drop, or `nullopt` when every
/// entry is pinned.
///
/// `nullopt` is a legitimate answer, not an error: the caller serves the
/// program uncached and pays the recompile next fire. A full cache must never
/// fail the work.
inline std::optional<std::uint64_t> pick_lru_victim(
    const std::vector<ProgramCacheEntry>& entries) {
    std::optional<std::uint64_t> victim;
    std::uint64_t oldest = 0;
    bool seen = false;
    for (const auto& entry : entries) {
        if (entry.use_count != 1) {
            continue;
        }
        if (!seen || entry.used_at < oldest) {
            oldest = entry.used_at;
            victim = entry.hash;
            seen = true;
        }
    }
    return victim;
}

}  // namespace pie::metal::pipeline
