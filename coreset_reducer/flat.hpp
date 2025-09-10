#pragma once

#include "coreset_reducer.hpp"

#include "errors.hpp"
#include "parser.hpp"
#include "metrics.hpp"

#include <random>
#include <span>
#include <vector>

namespace streamkm
{
  
template <std::random_device::result_type Seed = 0U, 
    std::uniform_random_bit_generator RandEng = std::mt19937, 
    typename Metrics = CoresetReducerNoMetrics>
struct FlatCoresetReducer
{    

    struct Result
    {
        size_t d, k;
        std::vector<float> points;
        Metrics::MetricsSet metrics;
    };

    static inline RandEng make_rand_eng()
    {
        if constexpr (Seed != 0U)
            return RandEng(Seed);
        else
            return RandEng(std::random_device{}());
    }

    template <bool with_weights>
    static inline double points_distance(const float *data, const size_t D, size_t a, size_t b)
    {
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
        std::size_t choosen_k = 1;
        
        Metrics metrics;
        metrics.start_init();
        
        auto compute_costs = [&data, &d, &n](size_t index, const std::vector<size_t> &indexes)
        {
            double total_cost = 0.0;
            for (size_t i : indexes)
                total_cost += points_distance<with_weights>(data, d, i, index);
            
                return total_cost;
        };

        RandEng rand_eng = make_rand_eng();
        std::uniform_real_distribution<double> rand_double(0.0, 1.0);
        std::uniform_int_distribution<size_t> rand_index(0, n - 1);

        double                              total_cost = 0.0;
        std::vector<double>                 cluster_costs;
        std::vector<size_t>                 cluster_centers;
        std::vector<std::vector<size_t>>    clusters;

        { // Root initialization
            size_t random_point = rand_index(rand_eng);
            std::vector<size_t> all_points(n);
            std::iota(all_points.begin(), all_points.end(), 0);

            double cost = compute_costs(random_point, all_points);

            cluster_costs.push_back(cost);
            cluster_centers.push_back(random_point);
            clusters.push_back(std::move(all_points));
            
            total_cost += cost;
        }

        metrics.end_init();

        while (choosen_k < k)   
        {   

            // ===================
            // Picking Leaf Node
            // ===================

            metrics.start_node_pick();
            size_t picked_cluster_id;
            {
                double random_double = rand_double(rand_eng) * total_cost;
                double cumulative = 0.0;
                for (size_t i = 0; i < cluster_costs.size(); i++)
                {
                    cumulative += cluster_costs[i];
                    if (cumulative >= random_double)
                    {
                        picked_cluster_id = i;
                        break;
                    }
                }

                passert(picked_cluster_id < cluster_costs.size(), "Cluster id out of range, got {}, max {}", picked_cluster_id, cluster_costs.size());

            }

            const size_t cluster_id = picked_cluster_id;
            const size_t cluster_center_idx = cluster_centers[cluster_id];
            const double cluster_curr_cost = cluster_costs[cluster_id];
            const std::vector<size_t> &cluster_points = clusters[cluster_id];
      
            metrics.end_node_pick();


            // ===================
            // Picking new center
            // ===================  

            metrics.start_new_center();

            double min_cost_center = std::numeric_limits<double>::max();
            size_t best_index = std::numeric_limits<size_t>::max();
            
            for (size_t round = 0; round < 3; round++)
            {

                size_t random_point = cluster_center_idx;
                double prob = rand_double(rand_eng) * cluster_curr_cost;
                double cumulative = 0.0;
                
                for (size_t i : cluster_points)
                {
                    cumulative += points_distance<with_weights>(data, d, i, cluster_center_idx);
                    if (cumulative >= prob)
                    {
                        random_point = i;
                        break;
                    }
                }

                // std::cout << cluster_curr_cost << " " << prob << " " << cumulative << " " << random_point << " " << cluster_center_idx << std::endl;

                passert(random_point != cluster_center_idx, "Random point should be different from current center {}, iteration {}, round {}", cluster_center_idx, choosen_k, round);


                // double cost = compute_costs(random_point, cluster_points);
                for (size_t i : cluster_points) {
                    double old_dist = points_distance<with_weights>(data, d, i, cluster_center_idx);
                    double new_dist = points_distance<with_weights>(data, d, i, random_point);
                    cost += std::min(old_dist, new_dist);
                }

                if (cost < min_cost_center)
                {
                    min_cost_center = cost;
                    best_index = random_point;
                }
            }

            metrics.end_new_center();

            // ===================
            // Splitting Node cluster
            // ===================  

            metrics.start_split();

            std::vector<size_t> left_points, right_points;
            for (size_t i : cluster_points)
            {   
                double dist_to_current = points_distance<with_weights>(data, d, i, cluster_center_idx);
                double dist_to_new = points_distance<with_weights>(data, d, i, best_index);

                if (dist_to_current < dist_to_new)    
                    left_points.push_back(i);
                else
                    right_points.push_back(i);
            }
            
            passert(!left_points.empty(), "Left cluster cannot be empty, iteration {}", choosen_k);
            passert(!right_points.empty(), "Right cluster cannot be empty, iteration {}", choosen_k);

            double cluster_new_cost = compute_costs(cluster_center_idx, left_points);
            double new_cluster_cost = compute_costs(best_index, right_points);

            // Update clusters
            total_cost -= cluster_curr_cost;
            total_cost += cluster_new_cost + new_cluster_cost;
            
            cluster_costs[cluster_id] = cluster_new_cost;
            cluster_centers[cluster_id] = cluster_center_idx;
            clusters[cluster_id] = std::move(left_points);

            cluster_costs.push_back(new_cluster_cost);
            cluster_centers.push_back(best_index);
            clusters.push_back(std::move(right_points));

            metrics.end_split();

            // ===================
            // Splitting Node cluster
            // ===================  
            
            metrics.start_cost_update();
            metrics.end_cost_update();

            metrics.end_iteration();
            choosen_k++;
        }   


        metrics.start_final_coreset();

        size_t stride = with_weights ? (size_t) (d + 1) : d;
        std::vector<float> centers(cluster_centers.size() * (d + 1), 0.0f);

        for (int cluster_id = 0; cluster_id < cluster_centers.size(); cluster_id++) {   
            size_t offs = cluster_id * (d + 1);
            size_t cidx = cluster_centers[cluster_id];

            double weight = 0.0;
            for (size_t pidx : clusters[cluster_id]) {
                float* p = &data[pidx * stride];

                if constexpr (with_weights) {
                    float w = p[d];
                    passert(w > 0.0f, "Weight should be positive, got {} for point {}", w, pidx);
                    weight += w;
                } else {
                    weight += 1.0f;
                }

                for (size_t i = 0; i < d; i++) {
                    centers[offs + i] += p[i];
                }

                passert(weight > 0.0f, "Weight should be positive, got {} for leaf center {} [with_weights={}]", weight, cidx, with_weights ? "true" : "false");

                centers[offs + d] = weight;
            }
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
};


static_assert(CoresetReducer<FlatCoresetReducer<0U, std::mt19937>>, "NaiveCoresetReducer does not satisfy CoresetReducer concept");

} // namespace streamkm
