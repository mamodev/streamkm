
#pragma once

#include "streamkm/core/random.hpp"
#include "streamkm/core/traits.hpp"

#include "streamkm/reducer/core/all.hpp"
#include "streamkm/reducer/core/reducer.hpp"

#include "streamkm/reducer/swap/core.hpp"

#include <span>
#include <vector>
#include <stack>

/*

    The main limiting factor of this coreset reducer algorithm is the intrinsic sequential nature.
    The seqeuential nature is due to the fact that picking a cluster to split depends on the cost of all clusters.

    This factor limits two main aspects:
    1) Parallelization: we cannot parallelize the splitting of clusters since each split depends on the global state of the cluster set.
    2) Memory access: we cannot access memory in a cache friendly manner since the cluster to split is arbitrary and could cause random memory access
         patterns, and thus cache thrashing.
    
    We need to leverage some heuristics to mitigate these issues.
    The main euristic we can find is that the tree is statically balanced.
*/

namespace streamkm
{
  
struct __PcSqNormsCachedSwapClusterProps {
    size_t start = 0;
    size_t size = 0;
    size_t split_count = 0;
 
    __PcSqNormsCachedSwapClusterProps(size_t start, size_t size, size_t split_count)
        : start(start), size(size), split_count(split_count) {}

     __PcSqNormsCachedSwapClusterProps() = default;
    // Make it movable but not copyable
    __PcSqNormsCachedSwapClusterProps(const __PcSqNormsCachedSwapClusterProps&) = delete;
    __PcSqNormsCachedSwapClusterProps(__PcSqNormsCachedSwapClusterProps&&) = default;
    __PcSqNormsCachedSwapClusterProps& operator=(const __PcSqNormsCachedSwapClusterProps&) = delete;
    __PcSqNormsCachedSwapClusterProps& operator=(__PcSqNormsCachedSwapClusterProps&&) = default;

};

static_assert(MovableNotCopyable<__PcSqNormsCachedSwapClusterProps>, "__PcSqNormsCachedSwapClusterProps should be movable but not copyable");


template <typename Metrics, template <MovableNotCopyable> typename CCS>
requires CoresetClusterSetFamily<CCS>
class _SwapPcSqNormsCachedCoresetReducer
{    
public:
    using Result = FlatWPointsResult;
    using ClusterProps = __PcSqNormsCachedSwapClusterProps;

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

        auto move_point = [&](size_t from, size_t to) {
            if (from != to) {
                for (size_t i = 0; i < STRIDE; i++) {
                    std::swap(data[from * STRIDE + i], data[to * STRIDE + i]);
                }

                std::swap(dcache[from], dcache[to]);
                std::swap(squared_norms[from], squared_norms[to]);
            }
        };

        
        double initial_cost = 0.0;
        auto init_root = [&]() -> std::pair<ClusterProps, double> {
            std::uniform_int_distribution<size_t> rand_index(0, n - 1);
            std::size_t random_point = rand_index(rand_eng);

            dcache.resize(n, 0.0);

            squared_norms = std::move(
                flat_wpoints_l2_sqnorms<WithWeights>(data, n, d)
            );

            auto root = ClusterProps(0, n, 0);
            move_point(random_point, 0); // Keep the first point as center

            double initial_cost = 0.0;
            for (size_t i = 0; i < n; i++) {
                double dist = flat_wpoints_l2_distance<WithWeights>(data, d, i, 0);
                dcache[i] = dist;
                initial_cost += dist;
            }


            return std::make_pair(
                std::move(root),
                initial_cost
            );
        };

        // init_root();


        // ===================
        // Initialization
        // ===================
        mtrs.start_init();

        CCS<ClusterProps> cluster_set(k, init_root(), rand_eng);

        // CCS cluster_set =  CCS(k, init_root(), rand_eng);
        // mtrs.end_init();

        mtrs.set_initial_tree_cost(initial_cost);


        auto pick_new_centre = [&] (size_t start_idx, size_t length, double cluster_cost) -> size_t {

            passert(dcache[start_idx] == 0.0, "Current center should have zero distance to itself, got {}, start_idx {}, length {}", dcache[start_idx], start_idx, length);

            double min_cost_center = std::numeric_limits<double>::max();    
            size_t best_index = std::numeric_limits<size_t>::max();

            for (size_t round = 0; round < 3; round++)
            {
                size_t random_point = std::numeric_limits<size_t>::max();
                double prob = rand_double(rand_eng) * cluster_cost;       
                double cumulative = 0.0;

                for (size_t i = 1; i < length; i++)
                {
                    cumulative += dcache[start_idx + i];
                    if (cumulative >= prob)
                    {
                        random_point = start_idx + i;
                        break;
                    }
                }

                passert(random_point != std::numeric_limits<size_t>::max(), "Random point should be different from size_t::max, it={} round={} prob={} cumulative={} node_cost={} node_start={} node_length={}",
                        choosen_k, round, prob, cumulative, cluster_cost, start_idx, length);

                double cost = 0.0;
                for (size_t i = 0; i < length; i++) {
                    double old_dist = dcache[start_idx + i];
                    double new_dist = flat_wpoints_l2_dot_distance<WithWeights>(data, squared_norms.data(), d, i, random_point);
                    cost += std::min(old_dist, new_dist);
                }

                if (cost < min_cost_center)
                {
                    min_cost_center = cost;
                    best_index = random_point;
                }
            }

            passert(best_index != std::numeric_limits<size_t>::max(), "Best index should have been set, got {} at iteration {}", best_index, choosen_k);
            return best_index;
        };

        auto partition_cluster = [&] (size_t start_idx, size_t length, size_t best_index) {
            passert(dcache[start_idx] == 0.0, "Current center should have zero distance to itself, got {}, start_idx {}, length {}", dcache[start_idx], start_idx, length);

            std::vector<bool> is_left_mask(length, false);

            double right_cost = 0.0, left_cost = 0.0;
            for (size_t i = 0; i < length; i++)
            {
                double dist_old_center = dcache[start_idx + i];
                double dist_new_center = flat_wpoints_l2_dot_distance<WithWeights>(data, squared_norms.data(), d, i + start_idx, best_index);
                
                bool take_left = dist_old_center <= dist_new_center;
                is_left_mask[i] = take_left;
                if (take_left) {
                    left_cost += dist_old_center;
                } else {
                    right_cost += dist_new_center;
                    dcache[start_idx + i] = dist_new_center; // Update dcache for the right child
                }
            }

            passert(!is_left_mask[best_index - start_idx], "Left mask should not include the new center point, start_idx {} length {} best_index {}, rel_best_index {}",
                    start_idx, length, best_index, best_index - start_idx);

            passert(is_left_mask[0], "Left mask should include the center point, iteration {}", choosen_k);

            size_t left_pos = 0;
            size_t right_pos = length - 1;

            while (left_pos <= right_pos) {
                if (is_left_mask[left_pos]) {
                    ++left_pos;
                    continue;
                }
                
                if (!is_left_mask[right_pos]) {
                    --right_pos;
                    continue;
                }

                bool tmp = is_left_mask[left_pos];
                is_left_mask[left_pos] = is_left_mask[right_pos];
                is_left_mask[right_pos] = tmp;

                move_point(left_pos + start_idx, right_pos + start_idx);

                if (best_index == left_pos + start_idx) best_index = right_pos + start_idx;
                else if (best_index == right_pos + start_idx) best_index = left_pos + start_idx;

                ++left_pos;
                --right_pos;
            }

            size_t left_count = left_pos;
            move_point(best_index, left_count + start_idx);


            return std::make_tuple(left_count, left_cost, right_cost);
        };


        // auto split_stack = std::stack<std::tuple<size_t, size_t, double>>(); // (start_idx, length, cluster_cost)
        // auto split_stack = std::stack<typename CCS<ClusterProps>::TCluster&>(); // (start_idx, length, cluster_ref)
        using TCluster = typename CCS<ClusterProps>::TCluster;
        auto split_stack = std::stack<TCluster*>(); // (start_idx, length, cluster_ref)


        split_stack.push(&(*cluster_set.begin())); // Ref to root

        const size_t split_threshold = 2 * (n / k); // Stop splitting when a cluster has less than n/k points
        const double balance_threshold = 0.05;


        size_t iters = 0;
        while ( !split_stack.empty() ) {

             if (iters > k - 4) {
                break;
            }

            // mtrs.start_node_pick();
            TCluster* cluster_ref = split_stack.top();
            size_t cluster_start = cluster_ref->props.start;
            size_t cluster_length = cluster_ref->props.size;
            double cluster_cost = cluster_ref->cost();
            split_stack.pop();
            // mtrs.end_node_pick();

            // mtrs.start_new_center();
            size_t best_index = pick_new_centre(cluster_start, cluster_length, cluster_cost);
            // mtrs.end_new_center();

            // mtrs.start_split();
            auto [left_count, left_cost, right_cost] = partition_cluster(cluster_start, cluster_length, best_index);
            passert(left_count > 0 && left_count < cluster_length, "Left count should be in (0, {}), got {}, start_idx {}, best_index {}", cluster_length, left_count, cluster_start, best_index);
            passert(left_cost >= 0.0, "Left cost should be positive, got {}, start_idx {}, best_index {} left_count {}", left_cost, cluster_start, best_index, left_count);
            passert(right_cost >= 0.0, "Right cost should be positive, got {}, start_idx {}, best_index {} right_count {}", right_cost, cluster_start, best_index, cluster_length - left_count);
            passert(dcache[cluster_start] == 0.0, "Current center should have zero distance to itself, got {}, start_idx {}, length {}", dcache[cluster_start], cluster_start, cluster_length);
            passert(dcache[cluster_start + left_count] == 0.0, "New center should have zero distance to itself, got {}, start_idx {}, length {}, left_count {}, best_index {}", dcache[cluster_start + left_count], cluster_start, cluster_length, left_count, best_index);
            // mtrs.end_split();

            // mtrs.start_cost_update();
            size_t right_count = cluster_length - left_count;
            double balance_ratio =
                static_cast<double>(std::min(left_count, right_count)) /
                static_cast<double>(std::max(left_count, right_count));

            // bool balanced_enough = (balance_ratio >= balance_threshold);
            // bool reached_threshold = (left_count <= split_threshold) || (right_count <= split_threshold);

           

            std::pair<TCluster&, TCluster&> new_clusters = cluster_set.cluster_split(
                std::ref(*cluster_ref),
                left_cost,
                right_cost,
                ClusterProps(cluster_start, left_count, cluster_ref->props.split_count + 1),
                ClusterProps(cluster_start + left_count, right_count, cluster_ref->props.split_count + 1)
            );

            if (cluster_ref->props.split_count < 16) {

                if (left_count > split_threshold) 
                    split_stack.push(&new_clusters.first);
                if (right_count > split_threshold) 
                    split_stack.push(&new_clusters.second);
            }

            // // if (!reached_threshold && balanced_enough) { // Continue blind splitting
            //     split_stack.push(&new_clusters.first);
            //     split_stack.push(&new_clusters.second);
            // }

            // mtrs.end_cost_update();

            iters++;

            // mtrs.end_iteration();
        }


        mtrs.end_init();
        
        choosen_k = cluster_set.size();

        std::cout << std::format("Finished splitting after {} iterations, got {} clusters / {} requested\n", iters, choosen_k, k);

        passert(choosen_k <= k, "Number of clusters should not exceed k, got {} > {}", choosen_k, k);
        while (choosen_k < k) {  // Sequential splitting
            mtrs.start_node_pick();
            auto& cluster_ref = cluster_set.pick();
            auto& cluster = cluster_ref.props;
            mtrs.end_node_pick();

            mtrs.start_new_center();
            size_t best_index = pick_new_centre(cluster.start, cluster.size, cluster_ref.cost());
            mtrs.end_new_center();

            mtrs.start_split();
            auto [left_count, left_cost, right_cost] = partition_cluster(cluster.start, cluster.size, best_index);
            mtrs.end_split();

            mtrs.start_cost_update();
            cluster_set.cluster_split(
                cluster_ref,
                left_cost,
                right_cost,
                ClusterProps(cluster.start, left_count, cluster.size + 1),
                ClusterProps(cluster.start + left_count, cluster.size - left_count, cluster.size + 1)
            );
            mtrs.end_cost_update();

            choosen_k++;
            mtrs.end_iteration();
        }


        mtrs.start_final_coreset();
        // auto centers = swap_coreset_reducer_extract<WithWeights, ClusterProps>(cluster_set, data, d, k);
        std::vector<float> centers = std::vector<float>(k * (d + 1), 0.0); // last dimension is weight
        size_t leaf_count = 0;
        
        for (const auto& cluster_ref : cluster_set) {
            ClusterProps const &cluster = cluster_ref.props;
            const size_t npoints = cluster.size;
            
            float *const center = centers.data() + leaf_count * (d + 1);
            double weight = 0.0;

            for (size_t i = 0; i < npoints; i++) {
                const float *const p = &data[(cluster.start + i) * STRIDE];

                if constexpr (WithWeights) 
                    weight += p[d];
                else 
                    weight += 1.0f;

                for (size_t j = 0; j < d; j++) 
                    center[j] += p[j];
            }

            passert(weight > 0.0, "Cluster weight should be positive, got {} for leaf center {} [WithWeights={}]", weight, leaf_count, WithWeights ? "true" : "false");

            center[d] = weight;
            leaf_count++;
        }


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


static_assert(CoresetReducer<_SwapPcSqNormsCachedCoresetReducer<CoresetReducerNoMetrics, FenwickCoresetClusterSet>>, "_SwapPcSqNormsCachedCoresetReducer does not satisfy CoresetReducer concept");

} // namespace streamkm
