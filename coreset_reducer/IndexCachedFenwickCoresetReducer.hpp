#pragma once

/*
    Improvements over v2:
    - Use better new better picking heuristic
*/


#include "core/all.hpp"
#include "coreset_reducer.hpp"
#include "metrics.hpp"

#include <random>
#include <span>
#include <vector>

namespace streamkm
{
  
template <std::random_device::result_type Seed = 0U, 
    std::uniform_random_bit_generator RandEng = std::mt19937, 
    typename Metrics = CoresetReducerNoMetrics>
struct IndexCachedFenwickCoresetReducer
{    

    struct Result {
        size_t d, k;
        std::vector<float> points;
        Metrics::MetricsSet metrics;
    };
 
    struct Cluster {
        double cost = 0.0;
        size_t index = 0; 
        std::vector<size_t> points;
        inline size_t size() const { return points.size(); }
    };

    template <bool with_weights>
    static inline double points_distance(const float *data, const size_t D, size_t a, size_t b) {
        if (a == b) return 0.0;

        double dist = 0.0;

        if constexpr (with_weights) 
        {
            double wa = data[a * (D + 1) + D];
            double wb = data[b * (D + 1) + D];

            passert(wa > 0.0, "Weight should be positive, got {} for point {}", wa, a);
            passert(wb > 0.0, "Weight should be positive, got {} for point {}", wb, b);

            for (size_t i = 0; i < D; i++)
            {
                double diff = (data[a * (D + 1) + i] / wa) - (data[b * (D + 1) + i] / wb);
                dist += diff * diff;
            }

        } else {
            for (size_t i = 0; i < D; i++)
            {
                double diff = data[a * D + i] - data[b * D + i];
                dist += diff * diff;
            }
        }

        return dist;
    }


    template <bool with_weights>
    static inline Result __reduce(float *data, std::size_t n, std::size_t d, std::size_t k)
    {   
        passert(n < std::numeric_limits<size_t>::max(), "Too many points, cannot handle more than {} points", std::numeric_limits<size_t>::max());
       
        // ====================
        // Helper functions and constants
        // ====================
        constexpr size_t MinSplitIters = 3;
        const size_t STRIDE = with_weights ? (d + 1) : d;

        // ====================
        // Global state 
        // ====================

        Metrics metrics;

        RandEng rand_eng = make_rand_eng<Seed, RandEng>();
        std::uniform_real_distribution<double> rand_double(0.0, 1.0);
        std::uniform_int_distribution<size_t> rand_index(0, n - 1);
        std::vector<double> random_pick_probs(MinSplitIters, 0.0);
        std::vector<size_t> random_pick_indices(MinSplitIters, 0);
        std::size_t random_point = rand_index(rand_eng);
        std::size_t choosen_k = 1;

        std::vector<double> dcache(n, std::numeric_limits<double>::max());

        double total_cost = 0.0;
        FenwickTree<double, size_t> cluster_costs(k);
        std::vector<Cluster> clusters;
        clusters.reserve(k);
        
        
        clusters.push_back(Cluster{
            .cost = 0.0,
            .index = random_point,
            .points = std::vector<size_t>()
        });

        
        // ===================  
        // Initialization
        // ===================
        metrics.start_init();

        clusters[0].points.resize(n);
        std::iota(clusters[0].points.begin(), clusters[0].points.end(), 0);
        { // Compute initial distances
            double cost = 0.0;
            for (size_t i : clusters[0].points) {
                double dist = points_distance<with_weights>(data, d, i, clusters[0].index);
                dcache[i] = dist;
                cost += dist;
            }
            clusters[0].cost = cost;
            cluster_costs.update(1, cost);
            total_cost = cost;
        }

   
        metrics.end_init();
        metrics.set_initial_tree_cost(total_cost);
        while (choosen_k < k)   
        {   
            passert(clusters.size() == choosen_k, "Clusters size should be {}, got {}", choosen_k, clusters.size());

            // ===================
            // Picking Leaf Node
            // ===================
            metrics.set_tree_cost(total_cost);
            metrics.start_node_pick();
            
            size_t cluster_idx = cluster_costs.bin_search_ge(rand_double(rand_eng) * total_cost) - 1;
            passert(cluster_idx < clusters.size(), "Cluster index should be valid, got {} >= {}", cluster_idx, clusters.size());
            
            Cluster& cluster = clusters[cluster_idx];
            const size_t cluster_size = cluster.size();
            
            passert(cluster.cost >= 0.0, "Cost should be non-negative, got {}", cluster.cost);
            passert(cluster_size > 1, "Cluster should have more than one point, got {} for cluster [{}/{}], at iteration {}", cluster_size, cluster_idx, clusters.size(), choosen_k);
         
            metrics.end_node_pick();
            metrics.set_node_size(cluster_size);

            passert(cluster_size > 1, "Node should have more than 1 point, got {}", cluster_size);

            // ===================
            // Picking new center
            // ===================  
            metrics.start_new_center();
            
            size_t split_iters = std::min(MinSplitIters, cluster_size - 1);
            { // Initializing random picks
                for (size_t i = 0; i < split_iters; i++)
                    random_pick_probs[i] = rand_double(rand_eng) * cluster.cost;

                std::sort(random_pick_probs.begin(), random_pick_probs.end());

                int p_index = 0;
                double cumulative = 0.0;
        
                for (const auto& pi : cluster.points)
                {   
                    cumulative += dcache[pi];
                    if (cumulative >= random_pick_probs[p_index])
                    {
                        random_pick_indices[p_index] = pi;
                        p_index++;
                        if (p_index >= split_iters) break;
                    }
                }

                if (p_index < split_iters) {
                    std::uniform_int_distribution<size_t> rand_index(0, cluster_size - 1);
                    for (; p_index < split_iters; p_index++) {
                        size_t rp = rand_index(rand_eng);
                        while(rp == cluster.index) {
                            rp = rand_index(rand_eng);
                        }

                        random_pick_indices[p_index] = cluster.points[rp];
                    }
                }
            }

            double min_cost_center = std::numeric_limits<double>::max();
            size_t best_index = std::numeric_limits<size_t>::max();
            
            // check distance
            for (size_t round = 0; round < split_iters; round++)
            {

                size_t random_point = random_pick_indices[round];

                double cost = 0.0;
                for (const auto& i : cluster.points) {
                    double old_dist = dcache[i];
                    double new_dist = points_distance<with_weights>(data, d, i, random_point);
                    cost += std::min(old_dist, new_dist);
                }

                if (cost < min_cost_center)
                {
                    min_cost_center = cost;
                    best_index = random_point;
                }
            }

            passert(best_index != std::numeric_limits<size_t>::max(), "Best index should have been set, got {} at iteration {}", best_index, choosen_k);

            metrics.end_new_center();

            // ===================
            // Splitting Node cluster
            // ===================  

            metrics.start_split();

            std::vector<size_t> left_points, right_points;
            double left_cost = 0.0;
            double right_cost = 0.0;

            for (size_t i : cluster.points)
            {   
                double old_dist = dcache[i];
                double new_dist = points_distance<with_weights>(data, d, i, best_index);

                if (old_dist < new_dist) {
                    left_points.push_back(i);
                    left_cost += old_dist;
                } else {
                    right_points.push_back(i);
                    right_cost += new_dist;
                    dcache[i] = new_dist;
                }
            }

            passert(!left_points.empty(), "Left cluster cannot be empty, iteration {}", choosen_k);
            passert(!right_points.empty(), "Right cluster cannot be empty, iteration {}", choosen_k);
            passert(dcache[cluster.index] == 0.0, "Center point should have zero distance to itself, got {} for center {}", dcache[cluster.index], cluster.index);
            passert(dcache[best_index] == 0.0, "New center point should have zero distance to itself, got {} for center {}", dcache[best_index], best_index);

            double old_cost = cluster.cost;

            Cluster& new_cluster = clusters.emplace_back(Cluster{
                .cost = right_cost,
                .index = best_index,
                .points = std::move(right_points)
            });

            cluster = Cluster{
                .cost = left_cost,
                .index = cluster.index,
                .points = std::move(left_points)
            };

            passert(new_cluster.cost >= 0.0, "Cost should be non-negative, got {} on right child iteration {}", new_cluster.cost, choosen_k);
            passert(cluster.cost >= 0.0, "Cost should be non-negative, got {} on left child iteration {}", cluster.cost, choosen_k);

            metrics.end_split();

            // ===================
            // Splitting Node cluster
            // ===================  
            
            metrics.start_cost_update();

            cluster_costs.update(cluster_idx + 1, left_cost - old_cost);
            cluster_costs.update(clusters.size(), right_cost);
            total_cost += ((left_cost + right_cost) - old_cost);
            choosen_k++;

            metrics.end_cost_update();
            metrics.end_iteration();
        }   


        metrics.start_final_coreset();
        std::vector<float> centers = std::vector<float>(k * (d + 1), 0.0f);
        passert(clusters.size() == k, "Clusters size should be {}, got {}", k, clusters.size());

        float* centers_ptr = centers.data();
        for (Cluster& cluster : clusters) {
            const size_t csize = cluster.size();
            passert(csize > 0, "Cluster should have at least one point, got {}", csize);

            double weight_sum = 0.0;
            for (size_t pi : cluster.points) {
                const float* p = &data[pi * STRIDE];

                double w = 1.0f;
                if constexpr (with_weights) {
                    w = static_cast<double>(p[d]);
                    passert(w > 0.0f, "Weight should be positive, got {} for point {}", w, pi);
                }

                weight_sum += w;

                for (size_t i = 0; i < d; i++) {    
                    centers_ptr[i] += p[i];
                }
            }

            centers_ptr[d] = static_cast<float>(weight_sum);
            centers_ptr += (d + 1);

        }

        metrics.end_final_coreset();

        return Result{d, k, std::move(centers), typename Metrics::MetricsSet(std::move(metrics))};
    }

    static inline Result reduce(float *data, std::size_t n, std::size_t d, std::size_t k)
    {
        return __reduce<false>(data, n, d, k);
    }

    static inline Result reduce(Result const &a, Result const &b)
    {
        std::vector<float> merged;

        merged.insert(merged.end(), a.points.begin(), a.points.end());
        merged.insert(merged.end(), b.points.begin(), b.points.end());
        auto reduced = __reduce<true>(merged.data(), merged.size() / (a.d + 1), a.d, std::max(a.k, b.k));
        
        reduced.metrics.merge(typename Metrics::MetricsSet(std::move(a.metrics)));
        reduced.metrics.merge(typename Metrics::MetricsSet(std::move(b.metrics)));
        return reduced;

    }

    static inline std::vector<float> to_flat_points(Result const &r)
    {
        return compute_weighted_points(r.points.data(), r.d, r.points.size() / (r.d + 1));
    }
};


static_assert(CoresetReducer<IndexCachedFenwickCoresetReducer<0U, std::mt19937>>, "IndexCachedFenwickCoresetReducer does not satisfy CoresetReducer concept");

} // namespace streamkm
