#pragma once

#include <vector>
#include <concepts>
#include <random>

namespace streamkm
{

    template <typename T>
    concept CoresetReducer =
        requires(T t, float *data, std::size_t n, std::size_t d, std::size_t k, typename T::Result &a, typename T::Result &b) {
            typename T::Result;
            requires std::movable<typename T::Result>;

            { t.reduce(data, n, d, k) } -> std::same_as<typename T::Result>;
            { t.reduce(a, b) } -> std::same_as<typename T::Result>;
            { t.to_flat_points(a) } -> std::same_as<std::vector<float>>;
        };

    template <typename R>
    struct __h_reducer_result_t {
        using type = typename std::remove_cvref_t<R>::Result;
    };

    template <typename R>
    using reducer_result_t = typename __h_reducer_result_t<R>::type;

    //=========================
    // Helper functions
    //=========================

    template <std::random_device::result_type Seed = 0U, 
        std::uniform_random_bit_generator RandEng = std::mt19937>
    static inline RandEng make_rand_eng()
    {
        if constexpr (Seed != 0U)
            return RandEng(Seed);
        else
            return RandEng(std::random_device{}());
    }
}