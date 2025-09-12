#pragma once

#pragma once
#include <cstdint>
#include <limits>

namespace streamkm {
struct xorshift128plus {
    using result_type = std::uint64_t;

    // Constructors
    xorshift128plus() : xorshift128plus(0x9e3779b97f4a7c15ULL) {}
    explicit xorshift128plus(std::uint64_t seed) { seed_with_splitmix64(seed); }

    // URBG required: operator()
    result_type operator()() {
        uint64_t t = s0;
        const uint64_t s = s1;
        s0 = s;
        t ^= t << 23;
        t ^= t >> 17;
        t ^= s;
        t ^= s >> 26;
        s1 = t;
        return t + s;
    }

    // URBG required: min() and max()
    static constexpr result_type min() { return std::numeric_limits<result_type>::min(); }
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    // Optional: reseed
    void seed(std::uint64_t seedval) { seed_with_splitmix64(seedval); }

private:


    uint64_t s0 = 0, s1 = 0;

    static uint64_t splitmix64_step(uint64_t& x) {
        x += 0x9e3779b97f4a7c15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    void seed_with_splitmix64(uint64_t seed) {
        // Generate two distinct non-zero states
        uint64_t x = seed;
        s0 = splitmix64_step(x);
        s1 = splitmix64_step(x);
        // Avoid the forbidden all-zero state just in case
        if (s0 == 0 && s1 == 0) {
            s1 = 0x9e3779b97f4a7c15ULL;
        }
    }
};
} // namespace streamkm