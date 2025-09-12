#pragma once

#include <vector>

namespace streamkm
{

template<bool WithWeights, typename ClusterProps>
static inline std::vector<float> swap_coreset_reducer_extract(
    auto const &cluster_set, const float * const data, const std::size_t d, const std::size_t k
) {

    const size_t STRIDE = WithWeights ? (d + 1) : d;
    std::vector<float> centers(k * (d + 1), 0.0); // last dimension is weight    
    size_t leaf_count = 0;
    for (const auto& cluster_ref : cluster_set) {
        ClusterProps const &cluster = cluster_ref.props;
        const size_t npoints = cluster.data.size() / STRIDE;

        float *const center = centers.data() + leaf_count * (d + 1);
        double weight = 0.0;

        for (size_t i = 0; i < npoints; i++) {
            const float *const p = &cluster.data[i * STRIDE];

            if constexpr (WithWeights) 
                weight += p[d];
            else 
                weight += 1.0f;

            for (size_t j = 0; j < d; j++) 
                center[j] += p[j];
        }

        passert(weight > 0.0f, "Weight should be positive, got {} for leaf center {} [WithWeights={}]", weight, leaf_count, WithWeights ? "true" : "false");
        center[d] = weight;
        leaf_count++;
    }
    return centers;
}

}