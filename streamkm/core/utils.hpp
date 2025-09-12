#pragma once
#include <vector>
#include <functional>


template <typename Container, typename Projection>
auto kahan_sum(const Container& c, Projection proj) {
    using T = decltype(proj(*std::begin(c)));

    T s = T{};
    T comp = T{};

    for (const auto& elem : c) {
        T x = proj(elem);
        T y = x - comp;
        T t = s + y;
        comp = (t - s) - y;
        s = t;
    }

    return s;
}

template <typename Container>
auto kahan_sum(const Container& c) {
    using T = decltype(*std::begin(c));

    T s = T{};
    T comp = T{};

    for (const auto& x : c) {
        T y = x - comp;
        T t = s + y;
        comp = (t - s) - y;
        s = t;
    }

    return s;
}


template <typename T>
double relative_error(const T& original, const T& computed) {
    double denom = std::abs(original);

    if (denom == 0.0) {
        return std::abs(computed);
    }

    return std::abs(computed - original) / denom;
}


