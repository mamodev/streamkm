
#pragma once

#include "streamkm/core/random.hpp"
#include "streamkm/core/traits.hpp"

#include "streamkm/reducer/core/all.hpp"
#include "streamkm/reducer/core/reducer.hpp"

#include "streamkm/reducer/swap/core.hpp"

#include <span>
#include <vector>

namespace streamkm
{
  
struct __SqNormsCachedSwapClusterProps {
    std::span<float>    data;
    std::span<double>    sqnorms; // cached squared norms for each point
    std::span<double>   dcache; // cached distance to current center

    __SqNormsCachedSwapClusterProps(std::span<float> d, std::span<double> sn, std::span<double> dc) 
        : data(d), sqnorms(sn), dcache(dc) {}
   
     __SqNormsCachedSwapClusterProps() = default;
    // Make it movable but not copyable
    __SqNormsCachedSwapClusterProps(const __SqNormsCachedSwapClusterProps&) = delete;
    __SqNormsCachedSwapClusterProps(__SqNormsCachedSwapClusterProps&&) = default;
    __SqNormsCachedSwapClusterProps& operator=(const __SqNormsCachedSwapClusterProps&) = delete;
    __SqNormsCachedSwapClusterProps& operator=(__SqNormsCachedSwapClusterProps&&) = default;

};

static_assert(MovableNotCopyable<__SqNormsCachedSwapClusterProps>, "__SqNormsCachedSwapClusterProps should be movable but not copyable");


template <typename Metrics, template <MovableNotCopyable> typename CCS>
requires CoresetClusterSetFamily<CCS>
class _SwapSqNormsCachedCoresetReducer
{    
public:
    using Result = FlatWPointsResult;
    using ClusterProps = __SqNormsCachedSwapClusterProps;

    Metrics::MetricsSet metrics;

    template <bool WithWeights>
    inline Result __reduce(float *const data, const std::size_t n, const std::size_t d, const std::size_t k)
    {   
        passert(n < std::numeric_limits<size_t>::max(), "Too many points, cannot handle more than {} points", std::numeric_limits<size_t>::max());

        // ====================
        // Helper functions and constants
        // ====================
        constexpr size_t DEFAULT_CENTER_IDX = 0;
        const size_t STRIDE = WithWeights ? (d + 1) : d;

        auto move_point = [&](ClusterProps& node, size_t from, size_t to) {
            if (from != to) {
                for (size_t i = 0; i < STRIDE; i++) {
                    std::swap(node.data[from * STRIDE + i], node.data[to * STRIDE + i]);
                }

                std::swap(node.dcache[from], node.dcache[to]);
                std::swap(node.sqnorms[from], node.sqnorms[to]);
            }
        };

        auto cluster_size = [STRIDE] (const ClusterProps& cluster) {
            return cluster.data.size() / STRIDE;
        };

        // ====================
        // Global state 
        // ====================
        Metrics mtrs;

        URBG rand_eng = URBG(std::random_device{}());
        std::uniform_real_distribution<double> rand_double(0.0, 1.0);
        std::uniform_int_distribution<size_t> rand_index(0, n - 1);
        std::size_t choosen_k = 1;

        std::vector<double> squared_norms;
        std::vector<double> dcache;

        auto init_root = [&]() -> std::pair<ClusterProps, double> {
            std::uniform_int_distribution<size_t> rand_index(0, n - 1);
            std::size_t random_point = rand_index(rand_eng);

            dcache.resize(n, 0.0);

            squared_norms = std::move(
                flat_wpoints_l2_sqnorms<WithWeights>(data, n, d)
            );

            auto root = ClusterProps(
                                        std::span<float>(data, n * STRIDE),
                                        std::span<double>(squared_norms.data(), squared_norms.size()),
                                        std::span<double>(dcache.data(), dcache.size())
                                    );
            
            passert(cluster_size(root) == n, "Root node should have {} points, got {}", n, cluster_size(root));
            passert(dcache.size() == n, "Dcache size should be {}, got {}", n, dcache.size());
            passert(squared_norms.size() == n, "Squared norms size should be {}, got {}", n, squared_norms.size());

            move_point(root, random_point, 0); // Keep the first point as center

            double cost = 0.0;
            for (size_t i = 0; i < n; i++) {
                double dist = flat_wpoints_l2_distance<WithWeights>(data, d, i, 0);
                dcache[i] = dist;
                cost += dist;
            }


            return std::make_pair(
                std::move(root),
                cost
            );
        };


        // ===================
        // Initialization
        // ===================
        mtrs.start_init();
        CCS cluster_set =  CCS(k, init_root(), rand_eng);
        mtrs.end_init();

        mtrs.set_initial_tree_cost(cluster_set.total_cost());
        while (choosen_k < k)   
        {   

            // ===================
            // Picking Leaf Node
            // ===================
            mtrs.set_tree_cost(cluster_set.total_cost());
            mtrs.start_node_pick();

            passert(cluster_set.total_cost() > 0.0, "Total cost should be positive, got {} at iteration {}", cluster_set.total_cost(), choosen_k);
            
            auto& node_ref = cluster_set.pick();
            ClusterProps &node = node_ref.props;
            passert(!node.data.empty(), "Picked node should have points, got 0 points, iteration {}", choosen_k);
            
            mtrs.end_node_pick();
            mtrs.set_node_size(cluster_size(node));

            // ===================
            // Picking new center
            // ===================  

            mtrs.start_new_center();

            double min_cost_center = std::numeric_limits<double>::max();
            size_t best_index = std::numeric_limits<size_t>::max();
            
            // check distance
            for (size_t round = 0; round < 3; round++)
            {
                const float *node_data_ptr = node.data.data();
                size_t random_point = DEFAULT_CENTER_IDX;
                double prob = rand_double(rand_eng) * node_ref.cost();       
                double cumulative = 0.0;

                for (size_t i = 0; i < cluster_size(node); i++)
                {
                    cumulative += node.dcache[i];
                    if (cumulative >= prob)
                    {
                        random_point = i;
                        break;
                    }
                }

                passert(random_point != DEFAULT_CENTER_IDX, "Random point should be different from current center, it={} round={} prob={} cumulative={} node_cost={}",
                        choosen_k, round, prob, cumulative, node_ref.cost());

                double cost = 0.0;
                for (size_t i = 0; i < cluster_size(node); i++) {
                    double old_dist = node.dcache[i];
                    double new_dist = flat_wpoints_l2_dot_distance<WithWeights>(node_data_ptr, node.sqnorms.data(), d, i, random_point);
                    cost += std::min(old_dist, new_dist);
                }

                if (cost < min_cost_center)
                {
                    min_cost_center = cost;
                    best_index = random_point;
                }
            }

            passert(best_index != std::numeric_limits<size_t>::max(), "Best index should have been set, got {} at iteration {}", best_index, choosen_k);


            mtrs.end_new_center();

            // ===================
            // Splitting Node cluster
            // ===================  

            mtrs.start_split();

            std::vector<bool> is_left_mask(cluster_size(node), false);

            double right_cost = 0.0, left_cost = 0.0;
            for (size_t i = 0; i < cluster_size(node); i++)
            {
                const float* data_ptr = node.data.data();
                double dist_old_center = node.dcache[i];
                double dist_new_center = flat_wpoints_l2_distance<WithWeights>(data_ptr, d, i, best_index);
                
                bool take_left = dist_old_center <= dist_new_center;
                is_left_mask[i] = take_left;
                if (take_left) {
                    left_cost += dist_old_center;
                } else {
                    right_cost += dist_new_center;
                    node.dcache[i] = dist_new_center; // Update dcache for the right child
                }
            }

            passert(is_left_mask[DEFAULT_CENTER_IDX], "Left mask should include the center point, iteration {}", choosen_k);
            passert(!is_left_mask[best_index], "Left mask should not include the new center point, iteration {}", choosen_k);

            size_t left_pos = 0;
            size_t right_pos = cluster_size(node) - 1;

            while (left_pos <= right_pos) {
                if (is_left_mask[left_pos]) {
                    ++left_pos;
                    continue;
                }
                
                if (!is_left_mask[right_pos]) {
                    --right_pos;
                    continue;
                }

                // std::swap(is_left_mask[left_pos], is_left_mask[right_pos]);
                bool tmp = is_left_mask[left_pos];
                is_left_mask[left_pos] = is_left_mask[right_pos];
                is_left_mask[right_pos] = tmp;

                move_point(node, left_pos, right_pos);

                if (best_index == left_pos) best_index = right_pos;
                else if (best_index == right_pos) best_index = left_pos;

                ++left_pos;
                --right_pos;
            }

            size_t left_count = left_pos;

            move_point(node, best_index, left_count);
            best_index = left_count;
   
            mtrs.end_split();

            // ===================
            // Splitting Node cluster
            // ===================  
            
            mtrs.start_cost_update();

            cluster_set.cluster_split(node_ref,
                left_cost,
                right_cost,
                ClusterProps(node.data.subspan(0, left_count * STRIDE),
                             node.sqnorms.subspan(0, left_count),
                             node.dcache.subspan(0, left_count)
                ),
                ClusterProps(node.data.subspan(left_count * STRIDE),
                             node.sqnorms.subspan(left_count),
                             node.dcache.subspan(left_count)
                )
            );

            choosen_k++;

            mtrs.end_cost_update();
            mtrs.end_iteration();
        }   


        mtrs.start_final_coreset();
        auto centers = swap_coreset_reducer_extract<WithWeights, ClusterProps>(cluster_set, data, d, k);
        mtrs.end_final_coreset();

        metrics.insert(std::move(mtrs));

        return Result{d, k, std::move(centers)};
    }

    inline Result reduce(float *data, std::size_t n, std::size_t d, std::size_t k)
    {
        return __reduce<false>(data, n, d, k);
    }

    inline Result reduce(Result const &a, Result const &b)
    {
        std::vector<float> merged;
        merged.insert(merged.end(), a.points.begin(), a.points.end());
        merged.insert(merged.end(), b.points.begin(), b.points.end());
        auto reduced = __reduce<true>(merged.data(), merged.size() / (a.d + 1), a.d, std::max(a.k, b.k));
        
        return reduced;
    }

    inline std::vector<float> to_flat_points(Result const &r)
    {
        return compute_weighted_points(r.points.data(), r.d, r.points.size() / (r.d + 1));
    }
};


static_assert(CoresetReducer<_SwapSqNormsCachedCoresetReducer<CoresetReducerNoMetrics, FenwickCoresetClusterSet>>, "_SwapSqNormsCachedCoresetReducer does not satisfy CoresetReducer concept");

} // namespace streamkm
