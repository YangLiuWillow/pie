// The program-cache eviction policy, tested without a Metal device.
//
// The cache this serves holds 64 compiled programs and used to REFUSE when
// full, which turned a capacity limit into a functional one: the fire never
// ran and the turn came back as a fluent empty completion. Eviction is what
// replaces that, so the two ways it can be silently wrong are what this pins.

#include "pipeline/program_cache_policy.hpp"

#include <cstdio>
#include <vector>

using pie::metal::pipeline::ProgramCacheEntry;
using pie::metal::pipeline::pick_lru_victim;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    // Plain LRU among entries the cache alone owns.
    {
        std::vector<ProgramCacheEntry> e{
            {0xA, 1, 30}, {0xB, 1, 10}, {0xC, 1, 20}};
        const auto v = pick_lru_victim(e);
        check(v.has_value() && *v == 0xB, "evicts the least recently used");
    }

    // THE one that must never be got wrong: a pinned entry is held by a live
    // instance or an in-flight fire. Evicting the map's reference is safe
    // (the owner keeps its own), but choosing a pinned entry over an
    // available one wastes the only eviction that would have freed capacity.
    {
        std::vector<ProgramCacheEntry> e{
            {0xA, 4, 1},   // oldest, but LIVE
            {0xB, 1, 50}};
        const auto v = pick_lru_victim(e);
        check(v.has_value() && *v == 0xB,
              "skips a pinned entry even when it is the oldest");
    }

    // Every entry live: declining is the correct answer, and the caller then
    // serves the program uncached rather than failing the fire.
    {
        std::vector<ProgramCacheEntry> e{{0xA, 2, 1}, {0xB, 3, 2}};
        check(!pick_lru_victim(e).has_value(),
              "declines when every entry is pinned");
    }

    // An empty cache has nothing to give, and must say so rather than
    // fabricating a hash.
    {
        std::vector<ProgramCacheEntry> e;
        check(!pick_lru_victim(e).has_value(), "declines on an empty cache");
    }

    // A tick of 0 is the oldest possible, not a missing value to be skipped:
    // an entry inserted but never hit again is exactly the right victim.
    {
        std::vector<ProgramCacheEntry> e{{0xA, 1, 0}, {0xB, 1, 7}};
        const auto v = pick_lru_victim(e);
        check(v.has_value() && *v == 0xA, "treats tick 0 as the oldest");
    }

    // A single unpinned entry is evictable; a cache of one must not deadlock
    // itself into serving everything uncached.
    {
        std::vector<ProgramCacheEntry> e{{0xA, 1, 5}};
        const auto v = pick_lru_victim(e);
        check(v.has_value() && *v == 0xA, "a lone unpinned entry is evictable");
    }

    if (failures == 0) {
        std::printf("program_cache_policy_test: all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
