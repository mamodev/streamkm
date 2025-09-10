#pragma once

#include "core/all.hpp"
#include "coreset_reducer.hpp"
#include "metrics.hpp"

#include <random>
#include <span>
#include <vector>

#include <cblas.h>


namespace streamkm
{
  
template <std::random_device::result_type Seed = 0U, 
    std::uniform_random_bit_generator RandEng = std::mt19937, 
    typename Metrics = CoresetReducerNoMetrics>
struct SwapCoresetReducer
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
        std::span<float> data;
        std::span<double> dcache;
        inline bool is_leaf() const { return lc == nullptr && rc == nullptr; }
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
        if (a == b) return dist;

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
    static inline Result __reduce(float *const data, const std::size_t n, const std::size_t d, const std::size_t k)
    {   
        passert(n < std::numeric_limits<size_t>::max(), "Too many points, cannot handle more than {} points", std::numeric_limits<size_t>::max());

        // ====================
        // Helper functions and constants
        // ====================
        constexpr size_t DEFAULT_CENTER_IDX = 0;
        const size_t STRIDE = with_weights ? (d + 1) : d;

        auto move_point = [&](Node& node, size_t from, size_t to) {
            if (from != to) {
                for (size_t i = 0; i < STRIDE; i++) {
                    std::swap(node.data[from * STRIDE + i], node.data[to * STRIDE + i]);
                }
                std::swap(node.dcache[from], node.dcache[to]);
            }
        };

        auto node_size = [d, STRIDE] (const Node* node) {
            return node->data.size() / STRIDE;
        };

        // ====================
        // Global state 
        // ====================
        Metrics metrics;

        RandEng rand_eng = make_rand_eng();
        std::uniform_real_distribution<double> rand_double(0.0, 1.0);
        std::uniform_int_distribution<size_t> rand_index(0, n - 1);
        std::size_t choosen_k = 1;

        std::vector<double> dcache(n, 0.0);

        Node root = {
            .lc = nullptr,
            .rc = nullptr,
            .parent = nullptr,
            .cost = 0.0,
            .data = std::span<float>(data, n * STRIDE),
            .dcache = std::span<double>(dcache)
        };
        passert(node_size(&root) == n, "Root node should have {} points, got {}", n, node_size(&root));

        // ===================
        // Initialization
        // ===================
        metrics.start_init();

        move_point(root, rand_index(rand_eng), 0); // Keep the first point as center
        for (size_t i = 0; i < node_size(&root); i++) {
            double cost = points_distance<with_weights>(root.data.data(), d, i, 0);
            root.dcache[i] = cost;
            root.cost += cost;
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
            passert(root.cost > 0.0, "Cost should be non-negative, got {}", root.cost);
            double random_double = rand_double(rand_eng);
            Node *node = &root;
            while (!node->is_leaf())
            {
                if (random_double < node->lc->cost / node->cost)
                    node = node->lc.get();
                else
                    node = node->rc.get();
            }
            metrics.end_node_pick();

            metrics.set_node_size(node_size(node));


            // ===================
            // Picking new center
            // ===================  

            metrics.start_new_center();

            double min_cost_center = std::numeric_limits<double>::max();
            size_t best_index = std::numeric_limits<size_t>::max();
            
            // check distance
            for (size_t round = 0; round < 3; round++)
            {

                size_t random_point = DEFAULT_CENTER_IDX;
                double prob = rand_double(rand_eng) * node->cost;       
                double cumulative = 0.0;
                const float *node_data_ptr = node->data.data();

                for (size_t i = 0; i < node_size(node); i++)
                {
                    cumulative += node->dcache[i];
                    if (cumulative >= prob)
                    {
                        random_point = i;
                        break;
                    }
                }

                passert(random_point != DEFAULT_CENTER_IDX, "Random point should be different from current center, it={} round={} prob={} cumulative={} node_cost={}",
                        choosen_k, round, prob, cumulative, node->cost);

                // double cost = compute_costs(node->data, random_point);
                double cost = 0.0;
                for (size_t i = 0; i < node_size(node); i++) {
                    // double old_dist = points_distance<with_weights>(node_data_ptr, d, i, DEFAULT_CENTER_IDX);
                    double old_dist = node->dcache[i];
                    double new_dist = points_distance<with_weights>(node_data_ptr, d, i, random_point);
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

            // std::vector<size_t> left_points;
            std::vector<bool> is_left_mask(node_size(node), false);

            double right_cost = 0.0, left_cost = 0.0;
            for (size_t i = 0; i < node_size(node); i++)
            {
                const float* data_ptr = node->data.data();
                double dist_old_center = node->dcache[i];
                double dist_new_center = points_distance<with_weights>(data_ptr, d, i, best_index);
                bool take_left = dist_old_center <= dist_new_center;
                is_left_mask[i] = take_left;

                if (take_left) {
                    left_cost += dist_old_center;
                } else {
                    right_cost += dist_new_center;
                    node->dcache[i] = dist_new_center; // Update dcache for the right child
                }
            }

            passert(is_left_mask[DEFAULT_CENTER_IDX], "Left mask should include the center point, iteration {}", choosen_k);
            passert(!is_left_mask[best_index], "Left mask should not include the new center point, iteration {}", choosen_k);

            size_t left_pos = 0;
            size_t right_pos = node_size(node) - 1;

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

                move_point(*node, left_pos, right_pos);

                if (best_index == left_pos) best_index = right_pos;
                else if (best_index == right_pos) best_index = left_pos;

                ++left_pos;
                --right_pos;
            }

            size_t left_count = left_pos;

            move_point(*node, best_index, left_count);
            best_index = left_count;

            
            node->lc = std::make_unique<Node>(Node{
                .lc = nullptr,
                .rc = nullptr,
                .parent = node,
                .cost = left_cost,
                .data = node->data.subspan(0, left_count * STRIDE),
                .dcache = node->dcache.subspan(0, left_count)
            });

            node->rc = std::make_unique<Node>(Node{
                .lc = nullptr,
                .rc = nullptr,
                .parent = node,
                .cost = right_cost,
                .data = node->data.subspan(left_count * STRIDE),
                .dcache = node->dcache.subspan(left_count)
            });

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
                size_t offs = centers.size();
                for (size_t i = 0; i < d; i++) {
                    centers.push_back(node->data[DEFAULT_CENTER_IDX * STRIDE + i]);
                }

                // sum with all points in the leaf
                float weight = 0;
                // Avoid double-adding the chosen center by iterating from 1
                for (size_t pi = 0; pi < node_size(node); pi++) {
                    const float* p = &node->data[pi * STRIDE];

                    if constexpr (with_weights) {
                        float w = p[d];
                        passert(w > 0.0f, "Weight should be positive, got {} for point {}", w, pi);

                        weight += w;
                    } else {
                        weight += 1.0f;
                    }

                    if (pi == DEFAULT_CENTER_IDX) continue; // Skip the center point, already added

                    
                    for (size_t i = 0; i < d; i++) {
                        centers[offs + i] += p[i];
                    }
                }

                passert(weight > 0.0f, "Weight should be positive, got {} for leaf center {} [with_weights={}]", 
                    weight, DEFAULT_CENTER_IDX, with_weights ? "true" : "false"); 

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

static_assert(CoresetReducer<SwapCoresetReducer<0U, std::mt19937>>, "SwapCoresetReducer does not satisfy CoresetReducer concept");

} // namespace streamkm
