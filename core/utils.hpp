#pragma once
#include <vector>
#include <functional>


namespace streamkm
{

template <typename T>
concept MovableNotCopyable = std::movable<T> && !std::copy_constructible<T>;


} // namespace streamkm

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


// layout: ROW-MAJOR (D + 1) x N   [x0,x1,...,xN-1, w0,w1,...,wN-1]
std::vector<float> compute_weighted_points(const float *wpoints, int D, int N) {
    std::vector<float> points(D * N);

    const int stride = D + 1;

    for (int i = 0; i < N; i++) {
        
        float w = wpoints[stride * i + D];


        for (int d = 0; d < D; d++) {
            points[i * D + d] = wpoints[stride * i + d] / w;
        }
    }

    return points;
}