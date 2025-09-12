#pragma once

#include "streamkm/core/random.hpp"
#include "streamkm/reducer/core/all.hpp"
#include "streamkm/reducer/indexed/core.hpp"

#include <vector>

namespace streamkm
{

template <typename Metrics, template <MovableNotCopyable> typename CCS>
requires CoresetClusterSetFamily<CCS>
struct _IndexSqNormsCachedCoresetReducer
{    
    Metrics::MetricsSet metrics;

    using Result = FlatWPointsResult;
    using ClusterProps = IndexCoresetReducerClusterProps;

    template <bool WithWeights>
    inline Result __reduce(float *data, std::size_t n, std::size_t d, std::size_t k) {   
        passert(n < std::numeric_limits<size_t>::max(), "Too many points, cannot handle more than {} points", std::numeric_limits<size_t>::max());
        Metrics mtrs;
        mtrs.start_init();

        std::vector<double> dist_cache(n, std::numeric_limits<double>::max());
        std::vector<double> squared_norms = flat_wpoints_l2_sqnorms<WithWeights>(data, n, d);

        auto compute_costs = [&](size_t index, const std::vector<size_t> &indexes)
        {
            double total_cost = 0.0;
            for (size_t i : indexes) {
                double cost = flat_wpoints_l2_distance<WithWeights>(data, d, i, index);
                dist_cache[i] = cost;
                total_cost += cost;
            }

            return total_cost;
        };

        URBG rand_eng = URBG(std::random_device{}());
        std::uniform_real_distribution<double> rand_double(0.0, 1.0);
        std::size_t choosen_k = 1;
        
        auto init_root = [&]() -> std::pair<ClusterProps, double> {
            std::uniform_int_distribution<size_t> rand_index(0, n - 1);
            std::size_t random_point = rand_index(rand_eng);
            std::vector<size_t> all_points(n);
            std::iota(all_points.begin(), all_points.end(), 0);
            double cost = compute_costs(random_point, all_points);
            return std::make_pair(
                ClusterProps(random_point, std::move(all_points)),
                cost
            );
        };

        auto cluster_set = CCS<ClusterProps>(k, init_root(), rand_eng);

        mtrs.end_init();

        mtrs.set_initial_tree_cost(cluster_set.total_cost());
        while (choosen_k < k)   
        {   
            // ===================
            // Picking Leaf Node
            // ===================

            mtrs.start_node_pick();

            passert(cluster_set.total_cost() > 0.0, "Total cost should be positive, got {} at iteration {}", cluster_set.total_cost(), choosen_k);
            
            auto& node_ref = cluster_set.pick();
            ClusterProps &node = node_ref.props;
            passert(!node.points.empty(), "Picked node should have points, got 0 points, iteration {}", choosen_k);

            mtrs.end_node_pick();
            mtrs.set_node_size(node.points.size());



            // ===================
            // Picking new center
            // ===================  
            passert(node.points.size() >= 2, "Cannot split a cluster with less than 2 points, got {} points at iteration {}", node.points.size(), choosen_k);

            mtrs.start_new_center();

            double min_cost_center = std::numeric_limits<double>::max();
            size_t best_index = std::numeric_limits<size_t>::max();
            
            // check distance
            for (size_t round = 0; round < 3; round++)
            {

                size_t random_point = node.index;
                double prob = rand_double(rand_eng) * node_ref.cost();       
                double cumulative = 0.0;
                
                for (size_t i : node.points)
                {
                    cumulative += flat_wpoints_l2_dot_distance<WithWeights>(data, squared_norms.data(), d, i, node.index);

                    if (cumulative >= prob)
                    {
                        random_point = i;
                        break;
                    }
                }

                passert(random_point != node.index, "Random point should be different from current center {}, iteration {}, round {}", node.index, choosen_k, round);


                double cost = 0.0;
                for (size_t i : node.points) {
                    double old_dist = dist_cache[i];
                    double new_dist = flat_wpoints_l2_dot_distance<WithWeights>(data, squared_norms.data(), d, i, random_point);
                    cost += std::min(old_dist, new_dist);
                }

                if (cost < min_cost_center)
                {
                    min_cost_center = cost;
                    best_index = random_point;
                }
            }

            mtrs.end_new_center();

            // ===================
            // Splitting Node cluster
            // ===================  

            mtrs.start_split();
            
            std::vector<size_t> left_points, right_points;
            double left_cost = 0.0, right_cost = 0.0;
            for (size_t i : node.points) {   
                double old_dist = dist_cache[i];
                double new_dist = flat_wpoints_l2_dot_distance<WithWeights>(data, squared_norms.data(), d, i, best_index);
                
                if (old_dist < new_dist) {
                    left_points.push_back(i);
                    left_cost += old_dist;
                } else {
                    right_points.push_back(i);
                    right_cost += new_dist;
                    dist_cache[i] = new_dist;
                }
            }

            mtrs.end_split();

            passert(!left_points.empty(), "Left cluster cannot be empty, iteration {}", choosen_k);
            passert(!right_points.empty(), "Right cluster cannot be empty, iteration {}", choosen_k);

             // ===================
            // Splitting Node cluster
            // ===================  
            
            mtrs.start_cost_update();

            cluster_set.cluster_split(node_ref,
                left_cost,
                right_cost,
                ClusterProps(node.index, std::move(left_points)),
                ClusterProps(best_index, std::move(right_points))
            );

            passert(node_ref.cost() >= 0.0, "Node cost should be non-negative after split, got {} at iteration {}", node_ref.cost(), choosen_k);

            choosen_k++;

            mtrs.end_cost_update();
            mtrs.end_iteration();
        }   

        mtrs.start_final_coreset();
        auto centers = index_coreset_reducer_extract<WithWeights>(cluster_set, data, d, k);
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


static_assert(CoresetReducer<_IndexSqNormsCachedCoresetReducer<CoresetReducerNoMetrics, FenwickCoresetClusterSet>>, "_IndexSqNormsCachedCoresetReducer does not satisfy CoresetReducer concept");

} // namespace streamkm
