#pragma once

#include "core/all.hpp"
#include "coreset_reducer.hpp"
#include "metrics.hpp"

#include <random>
#include <span>
#include <vector>

static size_t count = 0;

namespace streamkm
{
  
template <std::random_device::result_type Seed = 0U, 
    std::uniform_random_bit_generator RandEng = std::mt19937, 
    typename Metrics = CoresetReducerNoMetrics>

struct SwapFenwickCoresetReducer
{    

    struct Result
    {
        size_t d, k;
        std::vector<float> points;
        Metrics::MetricsSet metrics;
    };

    struct Cluster
    {
        double cost = 0.0;
        std::span<float> data;
        std::span<double> dcache;
        std::span<double> squared_norms;
    };

    static inline RandEng make_rand_eng()
    {
        if constexpr (Seed != 0U)
            return RandEng(Seed);
        else
            return RandEng(std::random_device{}());
    }

    template <bool WithWeights>
    static inline double points_distance(const float *const __restrict data, const double *const __restrict squared_norms, const size_t D, const size_t a, const size_t b)
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

    static inline void move_point(Cluster& cluster, const size_t from, const size_t to, const size_t stride) {
        for (size_t i = 0; i < stride; i++) 
            std::swap(cluster.data[from * stride + i], cluster.data[to * stride + i]);
        std::swap(cluster.dcache[from], cluster.dcache[to]);
        std::swap(cluster.squared_norms[from], cluster.squared_norms[to]);
    }

    template <bool with_weights>
    static inline Result __reduce(float *const data, const std::size_t n, const std::size_t d, const std::size_t k)
    {   

        // count++;
        // std::cout << count << "\n";
        passert(n < std::numeric_limits<size_t>::max(), "Too many points, cannot handle more than {} points", std::numeric_limits<size_t>::max());

        // ====================
        // Helper functions and constants
        // ====================
        constexpr size_t DEFAULT_CENTER_IDX = 0;
        constexpr size_t MinSplitIters = 3;
        const size_t STRIDE = with_weights ? (d + 1) : d;

        auto cluster_size = [d, STRIDE] (const Cluster& cluster) {
            return cluster.data.size() / STRIDE;
        };

        // ====================
        // Global state 
        // ====================
        std::vector<double> random_pick_probs(MinSplitIters, 0.0);
        std::vector<size_t> random_pick_indices(MinSplitIters, 0);

        Metrics metrics;

        RandEng rand_eng = make_rand_eng();
        std::uniform_real_distribution<double> rand_double(0.0, 1.0);
        std::uniform_int_distribution<size_t> rand_index(0, n - 1);
        std::size_t choosen_k = 1;

        double total_cost = 0.0;

        FenwickTree<double, size_t> cluster_costs(k);
        std::vector<double> dcache(n, 0.0);
        std::vector<double> squared_norms(n, 0.0);
        std::vector<bool>   is_left_mask(n, false);
        std::vector<double> tmp_dist(n, false);
        std::vector<double> tmp_dist2(n, false);
      
        std::vector<Cluster> clusters;
        clusters.reserve(k);

        clusters.push_back(Cluster{
            .cost = 0.0,
            .data = std::span<float>(data, n * STRIDE),
            .dcache = std::span<double>(dcache),
            .squared_norms = std::span<double>(squared_norms)
        });

        // ===================
        // Initialization
        // ===================
        metrics.start_init();


        for (size_t i = 0; i < n; ++i) {
            const float* pi = &data[i * STRIDE];
            // ||x||^2 = x.x = sum_j x_j^2
            double norm = 0.0;
            if constexpr (with_weights) {
                double w = pi[d];
                passert(w > 0.0, "Weight should be positive, got {} for point {}", w, i);
                for (size_t j = 0; j < d; ++j) {
                    double v = pi[j] / w;
                    norm += v * v;
                }
            } else {
                for (size_t j = 0; j < d; ++j) {
                    double v = pi[j];
                    norm += v * v;
                }
            }

            squared_norms[i] = norm;
        }
        
        move_point(clusters[0], rand_index(rand_eng), 0, STRIDE); // Keep the first point as center

        for (size_t i = 0; i < cluster_size(clusters[0]); i++) {
            double cost = points_distance<with_weights>(clusters[0].data.data(), squared_norms.data(), d, i, 0);
            clusters[0].dcache[i] = cost;
            clusters[0].cost += cost;
        }

        cluster_costs.update(1, clusters[0].cost);
        metrics.set_initial_tree_cost(clusters[0].cost);
        total_cost = clusters[0].cost;
       
        metrics.end_init();
        
        while (choosen_k < k)
        {   
            passert(clusters.size() == choosen_k, "Clusters size should be {}, got {}", choosen_k, clusters.size());
   
            // ===================
            // Picking Leaf Node
            // ===================

            metrics.start_node_pick();
            // double total_cost = cluster_costs.query(clusters.size());
            metrics.set_tree_cost(total_cost);
            
            size_t cluster_idx = cluster_costs.bin_search_ge(rand_double(rand_eng) * total_cost) - 1;
            passert(cluster_idx < clusters.size(), "Cluster index should be valid, got {} >= {}", cluster_idx, clusters.size());
            
            Cluster& cluster = clusters[cluster_idx];
            
            passert(cluster.cost >= 0.0, "Cost should be non-negative, got {}", cluster.cost);
            passert(cluster_size(cluster) > 1, "Cluster should have more than one point, got {} for cluster [{}/{}], at iteration {}", cluster_size(cluster), cluster_idx, clusters.size(), choosen_k);
         
            metrics.end_node_pick();

            metrics.set_node_size(cluster_size(cluster));

            // ===================
            // Picking new center
            // ===================  

            metrics.start_new_center();

            size_t split_iters = std::min(MinSplitIters, cluster_size(cluster) - 1);
             { // Initializing random picks
                for (size_t i = 0; i < split_iters; i++)
                    random_pick_probs[i] = rand_double(rand_eng) * cluster.cost;

                std::sort(random_pick_probs.begin(), random_pick_probs.end());


                int p_index = 0;
                double cumulative = 0.0;
        
                for (size_t pi = 1; pi < cluster_size(cluster) && p_index < split_iters; pi++)
                {   
                    cumulative += cluster.dcache[pi];

                    if (cumulative >= random_pick_probs[p_index])
                    {
                        random_pick_indices[p_index] = pi;
                        p_index++;
                    }
                }

                if (p_index < split_iters) {
                    std::uniform_int_distribution<size_t> rand_index(1, cluster_size(cluster) - 1);
                    for (; p_index < split_iters; p_index++) {
                        size_t rp = rand_index(rand_eng);
                        random_pick_indices[p_index] = rp;
                    }
                }
            }

            double min_cost_center = std::numeric_limits<double>::max();
            size_t best_index = std::numeric_limits<size_t>::max();
            
            // check distance
            for (size_t round = 0; round < split_iters; round++)
            {
                size_t random_point = random_pick_indices[round];
                const float *cluster_data_ptr = cluster.data.data();

                double cost = 0.0;
                for (size_t i = 0; i < cluster_size(cluster); i++) {
                    double old_dist = cluster.dcache[i];
                    double new_dist = points_distance<with_weights>(cluster_data_ptr, cluster.squared_norms.data(), d, i, random_point);
                    
                    double c = std::min(old_dist, new_dist);
                    tmp_dist2[i] = c;
                    
                    cost += c;
                }

                if (cost < min_cost_center)
                {
                    std::swap(tmp_dist, tmp_dist2);
                    min_cost_center = cost;
                    best_index = random_point;
                }
            }

            passert(best_index != std::numeric_limits<size_t>::max(), "Best index should have been set, got {} at iteration {} node_size={}", best_index, choosen_k, cluster_size(cluster));
            metrics.end_new_center();

            // ===================
            // Splitting Node cluster
            // ===================  

            // metrics.start_split();

            metrics.start_cost_update();
            double right_cost = 0.0, left_cost = 0.0;
            for (size_t i = 0; i < cluster_size(cluster); i++)
            {
                double dist_old_center = cluster.dcache[i];
                double dist_new_center = tmp_dist[i];

                bool take_left = dist_old_center <= dist_new_center;
                is_left_mask[i] = take_left;

                if (take_left) {
                    left_cost += dist_old_center;
                } else {
                    right_cost += dist_new_center;
                    cluster.dcache[i] = dist_new_center; // Update dcache for the right child
                }
            }
            metrics.end_cost_update();

            passert(cluster.dcache[best_index] == 0.0, "New center point should have zero distance to itself, got {} at iteration {}", cluster.dcache[best_index], choosen_k);
            passert(cluster.dcache[DEFAULT_CENTER_IDX] == 0.0, "Old center point should have zero distance to itself, got {} at iteration {}", cluster.dcache[DEFAULT_CENTER_IDX], choosen_k);

            passert(is_left_mask[DEFAULT_CENTER_IDX], "Left mask should include the center point, iteration {}", choosen_k);
            passert(!is_left_mask[best_index], "Left mask should not include the new center point, iteration {}", choosen_k);

            passert(left_cost >= 0.0, "Cost should be non-negative, got {} on left child iteration {}", left_cost, choosen_k);
            passert(right_cost >= 0.0, "Cost should be non-negative, got {} on right child iteration {}", right_cost, choosen_k);
            metrics.start_split();

            size_t left_pos = 0;
            size_t right_pos = cluster_size(cluster) - 1;

            while (left_pos <= right_pos) {
                if (is_left_mask[left_pos]) {
                    ++left_pos;
                    continue;
                }
                
                if (!is_left_mask[right_pos]) {
                    --right_pos;
                    continue;
                }

                move_point(cluster, left_pos, right_pos, STRIDE);

                if (best_index == left_pos) best_index = right_pos;
                else if (best_index == right_pos) best_index = left_pos;

                ++left_pos;
                --right_pos;
            }

            size_t left_count = left_pos;

            move_point(cluster, best_index, left_count, STRIDE);
            best_index = left_count;

            double old_cost = cluster.cost;

            clusters.push_back(Cluster{
                .cost = right_cost,
                .data = cluster.data.subspan(left_count * STRIDE),
                .dcache = cluster.dcache.subspan(left_count),
                .squared_norms = cluster.squared_norms.subspan(left_count)
            });


            cluster = Cluster{
                .cost = left_cost,
                // .data = node->data.subspan(0, left_count * STRIDE),
                .data = cluster.data.subspan(0, left_count * STRIDE),
                .dcache = cluster.dcache.subspan(0, left_count),
                .squared_norms = cluster.squared_norms.subspan(0, left_count)
            };

       

         
            metrics.end_split();

            // ===================
            // Splitting Node cluster
            // ===================  
            
            // metrics.start_cost_update();
            cluster_costs.update(cluster_idx + 1, left_cost - old_cost);
            cluster_costs.update(clusters.size(), right_cost);
            total_cost += ((left_cost + right_cost) - old_cost);
            choosen_k++;

            // metrics.end_cost_update();
            metrics.end_iteration();
        }   


        metrics.start_final_coreset();
        std::vector<float> centers = std::vector<float>(k * (d + 1), 0.0f);
        passert(clusters.size() == k, "Clusters size should be {}, got {}", k, clusters.size());

        float* centers_ptr = centers.data();

        for (Cluster& cluster : clusters) {
            const size_t csize = cluster_size(cluster);
            passert(csize > 0, "Cluster should have at least one point, got {}", csize);

            double weight_sum = 0.0;
            for (size_t pi = 0; pi < csize; pi++) {
                
                const float* p = &cluster.data[pi * STRIDE];

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

static_assert(CoresetReducer<SwapFenwickCoresetReducer<0U, std::mt19937>>, "SwapFenwickCoresetReducer does not satisfy CoresetReducer concept");

} // namespace streamkm
