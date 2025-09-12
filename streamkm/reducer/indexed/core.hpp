#pragma once

#include "streamkm/reducer/core/reducer.hpp"
#include "streamkm/core/traits.hpp"

namespace streamkm {

struct IndexCoresetReducerClusterProps
{
    size_t index = 0;
    std::vector<size_t> points;

    IndexCoresetReducerClusterProps() = default;
    IndexCoresetReducerClusterProps(size_t idx, std::vector<size_t>&& pts) : index(idx), points(std::move(pts)) {}
    
    // Make it movable but not copyable
    IndexCoresetReducerClusterProps(const IndexCoresetReducerClusterProps&) = delete;
    IndexCoresetReducerClusterProps(IndexCoresetReducerClusterProps&&) = default;
    IndexCoresetReducerClusterProps& operator=(IndexCoresetReducerClusterProps&&) = default;
    IndexCoresetReducerClusterProps& operator=(const IndexCoresetReducerClusterProps&) = delete;
};

static_assert(MovableNotCopyable<IndexCoresetReducerClusterProps>, "IndexCoresetReducerClusterProps should be movable but not copyable");


template<bool WithWeights>
static inline std::vector<float> index_coreset_reducer_extract(
    auto const &cluster_set, const float * const data, const std::size_t d, const std::size_t k
) {

    std::vector<float> centers(k * (d + 1), 0.0); // last dimension is weight
    size_t leaf_count = 0;
    for (const auto& cluster_ref : cluster_set) {
        IndexCoresetReducerClusterProps const &cluster = cluster_ref.props;
        passert(!cluster.points.empty(), "Cluster should have points, got 0 points {}", cluster.index);

        float *const center = centers.data() + leaf_count * (d + 1);
        double weight = 0.0;
        for (const size_t i : cluster.points) {
            const float *const p = &data[i * (WithWeights ? (d + 1) : d)];

            if constexpr (WithWeights) 
                weight += p[d];
            else 
                weight += 1.0f;

            for (size_t j = 0; j < d; j++) 
                center[j] += p[j];
        }

        passert(weight > 0.0f, "Weight should be positive, got {} for leaf center {} [WithWeights={}]", weight, cluster.index, WithWeights ? "true" : "false");
        center[d] = weight;
        leaf_count++;
    }

    return centers;
}



} // namespace streamkm

