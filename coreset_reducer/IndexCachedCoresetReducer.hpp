#pragma once

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
struct IndexCachedCoresetReducer
{    

    struct Result
    {
        size_t d, k;
        std::vector<float> points;
        Metrics::MetricsSet metrics;
    };

    struct Node
    {
        std::unique_ptr<Node> lc, rc;
        Node *parent = nullptr;
        double cost = 0.0;
        size_t index = 0;
        std::vector<size_t> points;
        bool is_leaf() const { return lc == nullptr && rc == nullptr; }
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

        std::size_t choosen_k = 1;
        std::size_t random_point = rand_index(rand_eng);

        std::vector<double> dist_cache(n, std::numeric_limits<double>::max());

        Node root = {
            .lc = nullptr,
            .rc = nullptr,
            .parent = nullptr,
            .cost = 0.0,
            .index = random_point,
            .points = std::vector<size_t>(n)
        };  

        std::iota(root.points.begin(), root.points.end(), 0);
        // root.cost = compute_costs(root.index, root.points);
        {
            double cost = 0.0;
            for (size_t i : root.points) {
                double dist = points_distance<with_weights>(data, d, i, root.index);
                dist_cache[i] = dist;
                cost += dist;
            }
            root.cost = cost;
        }
   
        metrics.end_init();

        metrics.set_initial_tree_cost(root.cost);
        while (choosen_k < k)   
        {   

            // ===================
            // Picking Leaf Node
            // ===================
            metrics.set_tree_cost(root.cost);
            metrics.start_node_pick();
            passert(root.cost >= 0.0, "Cost should be non-negative, got {}", root.cost);
            double random_double = rand_double(rand_eng);
            Node *node = &root;
            while (!node->is_leaf())
            {
                passert(node->cost > 0.0, "Cost should be positive, got {} on node at iteration {}", node->cost, choosen_k);
                if (random_double < node->lc->cost / node->cost)
                {
                    node = node->lc.get();
                }
                else
                {
                    node = node->rc.get();
                }
            }
            metrics.end_node_pick();

            metrics.set_node_size(node->points.size());

            // ===================
            // Picking new center
            // ===================  

            metrics.start_new_center();

            double min_cost_center = std::numeric_limits<double>::max();
            size_t best_index = std::numeric_limits<size_t>::max();
            
            // check distance
            for (size_t round = 0; round < 3; round++)
            {
                size_t random_point = node->index;
                double prob = rand_double(rand_eng) * node->cost;       
                double cumulative = 0.0;
                
                for (size_t i : node->points)
                {
                    // cumulative += points_distance<with_weights>(data, d, i, node->index);
                    cumulative += dist_cache[i];
                    if (cumulative >= prob)
                    {
                        random_point = i;
                        break;
                    }
                }

                passert(random_point != node->index, "Random point should be different from current center {}, iteration {}, round {}", node->index, choosen_k, round);


                // double cost = compute_costs(random_point, node->points);
                double cost = 0.0;
                for (size_t i : node->points) {
                    double old_dist = dist_cache[i];
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
            double left_cost = 0.0;
            double right_cost = 0.0;

            for (size_t i : node->points)
            {
                double old_dist = dist_cache[i];
                double new_dist = points_distance<with_weights>(data, d, i, best_index);

                if (old_dist < new_dist) {
                    left_points.push_back(i);
                    left_cost += old_dist;
                } else {
                    right_points.push_back(i);
                    right_cost += new_dist;
                    dist_cache[i] = new_dist;
                }
            }

            passert(!left_points.empty(), "Left cluster cannot be empty, iteration {}", choosen_k);
            passert(!right_points.empty(), "Right cluster cannot be empty, iteration {}", choosen_k);

            node->lc = std::make_unique<Node>(Node{
                .lc = nullptr,
                .rc = nullptr,
                .parent = node,
                .cost = left_cost,
                .index = node->index,
                .points = std::move(left_points)});

            node->rc = std::make_unique<Node>(Node{
                .lc = nullptr,
                .rc = nullptr,
                .parent = node,
                .cost = right_cost,
                .index = best_index,
                .points = std::move(right_points)});


            node->points.clear();
            node->points.shrink_to_fit();

            passert(node->lc->cost >= 0.0, "Cost should be non-negative, got {} on left child iteration {}", node->lc->cost, choosen_k);
            passert(node->rc->cost >= 0.0, "Cost should be non-negative, got {} on right child iteration {}", node->rc->cost, choosen_k);

            metrics.end_split();

            // ===================
            // Splitting Node cluster
            // ===================  
            
            metrics.start_cost_update();

            // Update parent cost
            Node *p = node;
            while (p != nullptr)
            {
                p->cost = p->lc->cost + p->rc->cost;
                p = p->parent;
            }

            choosen_k++;

            metrics.end_cost_update();
            metrics.end_iteration();
        }   


        metrics.start_final_coreset();
        std::vector<float> centers; 

        size_t leaf_size_acc = 0;
        size_t leaf_count = 0;

        std::vector<Node *> stack = {&root};
        while (!stack.empty())
        {
            Node *node = stack.back();
            stack.pop_back();
            if (node->is_leaf())
            {   
                size_t stride = with_weights ? (size_t) (d + 1) : d;
                size_t offs = centers.size();
                for (size_t i = 0; i < d; i++) {
                    centers.push_back(data[node->index * stride + i]);
                }

                leaf_size_acc += node->points.size();
                leaf_count += 1;

                // sum with all points in the leaf
                float weight = 0;
                for (size_t pi = 0; pi < node->points.size(); pi++) {
                    const size_t idx = node->points[pi];
                    const float* p = &data[idx * stride];

                    if constexpr (with_weights) {
                        float w = p[d];
                        passert(w > 0.0f,
                                "Weight should be positive, got {} for point {}",
                                w, idx);

                        weight += w;
                    } else {
                        weight += 1.0f;
                    }

                    // Avoid double-adding the chosen center
                    if (idx == node->index) continue;
                    
                    for (size_t i = 0; i < d; i++) {
                        centers[offs + i] += p[i];
                    }
                }

                passert(weight > 0.0f, "Weight should be positive, got {} for leaf center {} [with_weights={}]", weight, node->index, with_weights ? "true" : "false"); 
                centers.push_back(weight);

            }
            else
            {
                stack.push_back(node->lc.get());
                stack.push_back(node->rc.get());
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

    static inline std::vector<float> to_flat_points(Result const &r)
    {
        return compute_weighted_points(r.points.data(), r.d, r.points.size() / (r.d + 1));
    }
};


static_assert(CoresetReducer<IndexCachedCoresetReducer<0U, std::mt19937>>, "IndexCachedCoresetReducer does not satisfy CoresetReducer concept");

} // namespace streamkm
