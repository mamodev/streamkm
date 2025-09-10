#pragma once

/*
    Improvements over v1:
    - Use Arena allocation for nodes
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
struct IndexCachedCoresetReducer2
{    

    struct Result {
        size_t d, k;
        std::vector<float> points;
        Metrics::MetricsSet metrics;
    };
 
    struct Node {
        Node *parent = nullptr;
        Node *lc = nullptr;
        Node *rc = nullptr;
        double cost = 0.0;
        size_t index = 0; 
        std::vector<size_t> points;
        inline bool is_leaf() const { return lc == nullptr && rc == nullptr; }
    };

    struct NodeArena {
        explicit NodeArena(size_t capacity) : nodes(capacity) {}

        Node* alloc() {
            passert(size_ < nodes.size(), "Arena out of capacity: have {}, tried to alloc {}", nodes.size(), size_ + 1);
            return &nodes[size_++];
        }   
    
        private:
            std::vector<Node> nodes;
            size_t size_ = 0;
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
        auto compute_costs = [&data, &d, &n](size_t index, const std::vector<size_t> &indexes)
        {
            double total_cost = 0.0;
            for (size_t i : indexes)
                total_cost += points_distance<with_weights>(data, d, i, index);
            
                return total_cost;
        };


        // ====================
        // Global state 
        // ====================

        Metrics metrics;

        RandEng rand_eng = make_rand_eng<Seed, RandEng>();
        std::uniform_real_distribution<double> rand_double(0.0, 1.0);
        std::uniform_int_distribution<size_t> rand_index(0, n - 1);
        std::size_t choosen_k = 1;

        NodeArena arena(2 * k - 1);
        std::size_t random_point = rand_index(rand_eng);
        std::vector<double> dcache(n, std::numeric_limits<double>::max());

        Node* root = arena.alloc();
        *root = {
            .parent = nullptr,
            .lc = nullptr,
            .rc = nullptr,
            .cost = 0.0,
            .index = random_point,
            .points = std::vector<size_t>()
        };  
        
        // ===================  
        // Initialization
        // ===================
        metrics.start_init();
        root->points.resize(n);
        std::iota(root->points.begin(), root->points.end(), 0);

        // root.cost = compute_costs(root.index, root.points);
        {
            double cost = 0.0;
            for (size_t i : root->points) {
                double dist = points_distance<with_weights>(data, d, i, root->index);
                dcache[i] = dist;
                cost += dist;
            }
            root->cost = cost;
        }
   
        metrics.end_init();
        metrics.set_initial_tree_cost(root->cost);
        while (choosen_k < k)   
        {   

            // ===================
            // Picking Leaf Node
            // ===================
            metrics.set_tree_cost(root->cost);
            metrics.start_node_pick();
            
            passert(root->cost >= 0.0, "Cost should be non-negative, got {}", root->cost);
            double prob = rand_double(rand_eng);
            Node *node = root;
            
            while (!node->is_leaf())
                if (prob < node->lc->cost / node->cost)
                    node = node->lc;
                else
                    node = node->rc;

            metrics.end_node_pick();
            metrics.set_node_size(node->points.size());
            passert(node->points.size() > 1, "Node should have more than 1 point, got {}", node->points.size());

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
                
                bool found = false;
                for (size_t i : node->points)
                {   
                    cumulative += dcache[i];
                    if (cumulative >= prob)
                    {
                        found = true;
                        random_point = i;
                        break;
                    }
                }

                passert(random_point != node->index, 
                    "Random point should be different from current center {}, iteration {}, round {} node_size={} node_cost={} prob={} cumulative={} found={} center_cost={}",
                    node->index, choosen_k, round, node->points.size(), node->cost, prob, cumulative, found, dcache[node->index]
                );

                double cost = 0.0;
                for (size_t i : node->points) {
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
            passert(dcache[node->index] == 0.0, "Center point should have zero distance to itself, got {} for center {}", dcache[node->index], node->index);
            passert(dcache[best_index] == 0.0, "New center point should have zero distance to itself, got {} for center {}", dcache[best_index], best_index);

            Node *lc, *rc;
            lc = arena.alloc();
            rc = arena.alloc();
            *lc = Node{
                .parent = node,
                .lc = nullptr,
                .rc = nullptr,
                .cost = left_cost,
                .index = node->index,
                .points = std::move(left_points)
            };
            
            *rc = Node{
                .parent = node,
                .lc = nullptr,
                .rc = nullptr,
                .cost = right_cost,
                .index = best_index,
                .points = std::move(right_points)
            };

            node->lc = lc;
            node->rc = rc;
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

        std::vector<Node *> stack = {root};
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


static_assert(CoresetReducer<IndexCachedCoresetReducer2<0U, std::mt19937>>, "IndexCachedCoresetReducer2 does not satisfy CoresetReducer concept");

} // namespace streamkm
