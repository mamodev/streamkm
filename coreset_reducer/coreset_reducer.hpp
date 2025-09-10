#pragma once

#include <concepts>

namespace streamkm
{
    template <typename T>
    using h_rr_t = decltype(T::reduce(
        std::declval<float *>(),
        std::declval<std::size_t>(),
        std::declval<std::size_t>(),
        std::declval<std::size_t>()));

    template <typename T>
    concept CoresetReducer =
        std::movable<h_rr_t<T>> &&
        requires(float *data, std::size_t n, std::size_t d, std::size_t k, h_rr_t<T> const &a, h_rr_t<T> const &b) {
            { T::reduce(a, b) } -> std::same_as<h_rr_t<T>>;
            { T::reduce(data, n, d, k) } -> std::same_as<h_rr_t<T>>;
            { T::to_flat_points(a) } -> std::same_as<std::vector<float>>;
        };

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