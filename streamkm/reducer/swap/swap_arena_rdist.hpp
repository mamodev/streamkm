#pragma once

#include "core/all.hpp"
#include "coreset_reducer.hpp"
#include "metrics.hpp"

#include <random>
#include <span>
#include <vector>

/*
What changed?
- Using Norm2 chache and using dot product to compute distances
- Using Distance caching when computing best center to split
*/

namespace streamkm
{
  
template <std::random_device::result_type Seed = 0U, 
    std::uniform_random_bit_generator RandEng = std::mt19937, 
    typename Metrics = CoresetReducerNoMetrics>
struct SwapArenaRDistCoresetReducer
{    

    struct Result
    {
        size_t d, k;
        std::vector<float> points;
        Metrics::MetricsSet metrics;
    };

    struct Node
    {
        Node *parent = nullptr;
        Node *lc = nullptr;
        Node *rc = nullptr;
        double cost = 0.0;
        std::span<float> data;
        std::span<double> dcache;
        std::span<double> squared_norms;
        inline bool is_leaf() const { return lc == nullptr && rc == nullptr; }
    };

    struct NodeArena {

        explicit NodeArena(size_t capacity) : nodes(capacity) {}

        Node* alloc() {
            passert(size_ < nodes.size(), "Arena out of capacity: have {}, tried to alloc {}", nodes.size(), size_ + 1);
            return &nodes[size_++];
        }   

        template<typename... Args>
        Node* emplace_alloc(Args&&... args) {
            passert(size_ < nodes.size(), "Arena out of capacity: have {}, tried to alloc {}", nodes.size(), size_ + 1);
            nodes[size_] = Node{std::forward<Args>(args)...};
            return &nodes[size_++];
        }

    private:
        std::vector<Node> nodes;
        size_t size_ = 0;
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

    static inline void move_point(Node& node, const size_t from, const size_t to, const size_t stride) {
        for (size_t i = 0; i < stride; i++) {
            std::swap(node.data[from * stride + i], node.data[to * stride + i]);
        }
        std::swap(node.dcache[from], node.dcache[to]);
        std::swap(node.squared_norms[from], node.squared_norms[to]);
    }


    template <bool with_weights>
    static inline Result __reduce(float *const data, const std::size_t n, const std::size_t d, const std::size_t k)
    {   
        passert(n < std::numeric_limits<size_t>::max(), "Too many points, cannot handle more than {} points", std::numeric_limits<size_t>::max());

        // ====================
        // Helper functions and constants
        // ====================
        constexpr size_t DEFAULT_CENTER_IDX = 0;
        constexpr size_t MinSplitIters = 3;
        const size_t STRIDE = with_weights ? (d + 1) : d;

        auto node_size = [d, STRIDE] (const Node* node) {
            return node->data.size() / STRIDE;
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
        
        NodeArena arena(2 * k - 1);
        std::vector<double> dcache(n, 0.0);
        std::vector<double> squared_norms(n, 0.0);
        std::vector<bool> is_left_mask(n, false);
        std::vector<double> tmp_dist(n, false);
        std::vector<double> tmp_dist2(n, false);


        Node* root = arena.alloc();
        *root = Node{
            .parent = nullptr,
            .lc = nullptr,
            .rc = nullptr,
            .cost = 0.0,
            .data = std::span<float>(data, n * STRIDE),
            .dcache = std::span<double>(dcache),
            .squared_norms = std::span<double>(squared_norms)
        };
        passert(node_size(root) == n, "Root node should have {} points, got {}", n, node_size(root));

  

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
        
        move_point(*root, rand_index(rand_eng), 0, STRIDE); // Keep the first point as center

        for (size_t i = 0; i < node_size(root); i++) {
            double cost = points_distance<with_weights>(root->data.data(), squared_norms.data(), d, i, 0);
            root->dcache[i] = cost;
            root->cost += cost;
        }
       
        metrics.end_init();
        
        metrics.set_initial_tree_cost(root->cost);
        while (choosen_k < k)   
        {   

            // ===================
            // Picking Leaf Node
                // ===================

            metrics.start_node_pick();
            metrics.set_tree_cost(root->cost);
            passert(root->cost >= 0.0, "Cost should be non-negative, got {}", root->cost);
            double random_double = rand_double(rand_eng);
            Node *node = root;
            while (!node->is_leaf())
            {
                if (random_double < node->lc->cost / node->cost)
                    node = node->lc;
                else
                    node = node->rc;
            }
            metrics.end_node_pick();

            // passert(node->cost > 0.0, "Cost should be non-negative, got {} at iteration {}", node->cost, choosen_k);
            passert(node->is_leaf(), "Node should be a leaf, iteration {}", choosen_k);
            passert(node->cost > 0.0, "Cost should be greater than zero, got {} at iteration {}", node->cost, choosen_k);
            passert(node_size(node) > 1, "Node should have more than one point, got {} at iteration {} cost={}", node_size(node), choosen_k, node->cost);
            metrics.set_node_size(node_size(node));

            // ===================
            // Picking new center
            // ===================  

            metrics.start_new_center();

            size_t split_iters = std::min(MinSplitIters, node_size(node) - 1);
             { // Initializing random picks
                for (size_t i = 0; i < split_iters; i++)
                    random_pick_probs[i] = rand_double(rand_eng) * node->cost;

                std::sort(random_pick_probs.begin(), random_pick_probs.end());


                int p_index = 0;
                double cumulative = 0.0;
        
                for (size_t pi = 1; pi < node_size(node) && p_index < split_iters; pi++)
                {   
                    cumulative += node->dcache[pi];

                    if (cumulative >= random_pick_probs[p_index])
                    {
                        random_pick_indices[p_index] = pi;
                        p_index++;
                    }
                }

                if (p_index < split_iters) {
                    std::uniform_int_distribution<size_t> rand_index(1, node_size(node) - 1);
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
                const float *node_data_ptr = node->data.data();

                double cost = 0.0;
                for (size_t i = 0; i < node_size(node); i++) {
                    double old_dist = node->dcache[i];
                    double new_dist = points_distance<with_weights>(node_data_ptr, node->squared_norms.data(), d, i, random_point);
                    
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

            passert(best_index != std::numeric_limits<size_t>::max(), "Best index should have been set, got {} at iteration {} node_size={}", best_index, choosen_k, node_size(node));
            metrics.end_new_center();

            // ===================
            // Splitting Node cluster
            // ===================  

            metrics.start_split();

            // std::vector<size_t> left_points;

            double right_cost = 0.0, left_cost = 0.0;
            for (size_t i = 0; i < node_size(node); i++)
            {
                double dist_old_center = node->dcache[i];
                double dist_new_center = tmp_dist[i];

                bool take_left = dist_old_center <= dist_new_center;
                is_left_mask[i] = take_left;

                if (take_left) {
                    left_cost += dist_old_center;
                } else {
                    right_cost += dist_new_center;
                    node->dcache[i] = dist_new_center; // Update dcache for the right child
                }
            }

            passert(node->dcache[best_index] == 0.0, "New center point should have zero distance to itself, got {} at iteration {}", node->dcache[best_index], choosen_k);
            passert(node->dcache[DEFAULT_CENTER_IDX] == 0.0, "Old center point should have zero distance to itself, got {} at iteration {}", node->dcache[DEFAULT_CENTER_IDX], choosen_k);

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

                move_point(*node, left_pos, right_pos, STRIDE);

                if (best_index == left_pos) best_index = right_pos;
                else if (best_index == right_pos) best_index = left_pos;

                ++left_pos;
                --right_pos;
            }

            size_t left_count = left_pos;

            move_point(*node, best_index, left_count, STRIDE);
            best_index = left_count;


            Node* lc = arena.alloc();
            Node* rc = arena.alloc();

            *lc = Node{
                .parent = node,
                .lc = nullptr,
                .rc = nullptr,
                .cost = left_cost,
                .data = node->data.subspan(0, left_count * STRIDE),
                .dcache = node->dcache.subspan(0, left_count),
                .squared_norms = node->squared_norms.subspan(0, left_count)
            };

            *rc = Node{
                .parent = node,
                .lc = nullptr,
                .rc = nullptr,
                .cost = right_cost,
                .data = node->data.subspan(left_count * STRIDE),
                .dcache = node->dcache.subspan(left_count),
                .squared_norms = node->squared_norms.subspan(left_count)
            };
            
            node->lc = lc;
            node->rc = rc;
            
            passert(lc->cost >= 0.0, "Cost should be non-negative, got {} on left child iteration {}", lc->cost, choosen_k);
            passert(rc->cost >= 0.0, "Cost should be non-negative, got {} on right child iteration {}", rc->cost, choosen_k);

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

        std::vector<Node *> stack = {root};
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
                stack.push_back(node->lc);
                stack.push_back(node->rc);
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

static_assert(CoresetReducer<SwapArenaRDistCoresetReducer<0U, std::mt19937>>, "SwapArenaRDistCoresetReducer does not satisfy CoresetReducer concept");

} // namespace streamkm
