#pragma once

#include <cstddef>
#include <vector>

namespace streamkm {
    
    struct FlatWPointsResult
    {
        size_t d, k;
        std::vector<float> points;
    };

    template <bool WithWeights>
    static inline double flat_wpoints_l2_distance(const float *const __restrict data, const size_t D, const size_t a, const size_t b)
    {
        if (a == b) return 0.0;

        double dist = 0.0;

        if constexpr (WithWeights) {
            const double wa = data[a * (D + 1) + D];
            const double wb = data[b * (D + 1) + D];

            passert(wa > 0.0, "Weight should be positive, got {} for point {}", wa, a);
            passert(wb > 0.0, "Weight should be positive, got {} for point {}", wb, b);

            for (size_t i = 0; i < D; i++)
            {
                const double diff = (data[a * (D + 1) + i] / wa) - (data[b * (D + 1) + i] / wb);
                dist += diff * diff;
            }

        } else {
            for (size_t i = 0; i < D; i++)
            {
                const double diff = data[a * D + i] - data[b * D + i];
                dist += diff * diff;
            }
        }

        return dist;
    }

    template <bool WithWeights>
    static inline std::vector<double> flat_wpoints_l2_sqnorms(const float *const __restrict data,  const size_t N, const size_t D)
    {
        const size_t STRIDE = WithWeights ? (D + 1) : D;
        std::vector<double> squared_norms(N, 0.0);

        // ||x||^2 = x.x = sum_j x_j^2
        for (size_t i = 0; i < N; ++i) {
            const float* pi = &data[i * STRIDE];
            
            double norm = 0.0;
            if constexpr (WithWeights) {
                double w = pi[D];
                passert(w > 0.0, "Weight should be positive, got {} for point {}", w, i);
                for (size_t j = 0; j < D; ++j) {
                    double v = pi[j] / w;
                    norm += v * v;
                }
            } else {
                for (size_t j = 0; j < D; ++j) {
                    double v = pi[j];
                    norm += v * v;
                }
            }

            squared_norms[i] = norm;
        }

        return squared_norms;
    }


    template <bool WithWeights>
    static inline double flat_wpoints_l2_dot_distance(const float *const __restrict data, const double *const __restrict squared_norms, const size_t D, const size_t a, const size_t b)
    {
        if (a == b) return 0.0;

        const size_t STRIDE = WithWeights ? (D + 1) : D;
        double dot      = 0.0;
        double l2sum    = squared_norms[a] + squared_norms[b];

        const float * __restrict pa = &data[a * STRIDE];
        const float * __restrict pb = &data[b * STRIDE];
        
        for (size_t i = 0; i < D; i++)
                dot += pa[i] * pb[i];
        
        
        if constexpr (WithWeights) 
        {
            const double wa = data[a * STRIDE + D];
            const double wb = data[b * STRIDE + D];
            passert(wa > 0.0, "Weight should be positive, got {} for point {}", wa, a);
            passert(wb > 0.0, "Weight should be positive, got {} for point {}", wb, b);
            dot /= (wa * wb);
        } 
        
        return std::max(0.0, l2sum - 2.0 * dot);
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
}