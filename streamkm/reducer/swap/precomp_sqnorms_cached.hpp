
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

        const size_t L1Cache_Size = 32 * 1024; // 32 KB
        const size_t L2Cache_Size = 512 * 1024; // 256 KB
        const size_t Points_Per_L1 = L1Cache_Size / (STRIDE * sizeof(float)); // Number of points that fit in L1 cache
        const size_t Points_Per_L2 = L2Cache_Size / (STRIDE * sizeof(float)); // Number of points that fit in L2 cache



        // ====================
        // Global state 
        // ====================
        Metrics mtrs;

        URBG rand_eng = URBG(std::random_device{}());
        std::uniform_real_distribution<double> rand_double(0.0, 1.0);
        std::uniform_int_distribution<size_t> rand_index(0, n - 1);
        std::size_t choosen_k = 1;

        std::vector<size_t> indices(n, std::numeric_limits<size_t>::max());

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


        const size_t MinSplitIters = 3;
        std::vector<double> random_pick_probs(MinSplitIters, 0.0);
        std::vector<size_t> random_pick_indices(MinSplitIters, 0);

        auto pick_new_centre = [&] (size_t start_idx, size_t length, double cluster_cost) -> size_t {
            passert(dcache[start_idx] == 0.0, "Current center should have zero distance to itself, got {}, start_idx {}, length {}", dcache[start_idx], start_idx, length);

            size_t best_index = std::numeric_limits<size_t>::max();
            const size_t split_iters = std::min(MinSplitIters, length - 1);
            for (size_t i = 0; i < split_iters; i++)
                random_pick_probs[i] = rand_double(rand_eng) * cluster_cost;
            std::sort(random_pick_probs.begin(), random_pick_probs.end());
            

            if (length > Points_Per_L1) { // Use swap version

                { // Initializing random picks
                    size_t p_index = 0;
                    double cumulative = 0.0;
                    for (size_t i = 0; i < length; i++)
                    {
                        cumulative += dcache[start_idx + i];
                        if (cumulative >= random_pick_probs[p_index])
                        {
                            random_pick_indices[p_index] = start_idx + i;
                            p_index++;
                            if (p_index >= split_iters) break;
                        }
                    }
    
                    // ensure all picks are filled
                    if (p_index < split_iters) {
                        std::uniform_int_distribution<size_t> rand_index(1, length - 1);
                        for (; p_index < split_iters; p_index++) {
                            size_t rp = rand_index(rand_eng) + start_idx;
                            random_pick_indices[p_index] = rp;
                        }
                    }
                }
    
                // double min_cost_center = std::numeric_limits<double>::max();    
                // size_t best_index = std::numeric_limits<size_t>::max();
    
                // for (size_t round = 0; round < split_iters; round++)
                // {
                //     size_t random_point = random_pick_indices[round];
                //     double cost = 0.0;
                //     for (size_t i = 0; i < length; i++) {
                //         double old_dist = dcache[start_idx + i];
                //         double new_dist = flat_wpoints_l2_dot_distance<WithWeights>(data, squared_norms.data(), d, i, random_point);
                //         cost += std::min(old_dist, new_dist);
                //     }
    
                //     if (cost < min_cost_center)
                //     {
                //         min_cost_center = cost;
                //         best_index = random_point;
                //     }
                // }
    
                // let's create a parallel version where we compute all costs in parallel to exploit cache locality
    
                for (size_t round = 0; round < split_iters; round++) {
                    move_point(random_pick_indices[round], start_idx + 1 + round);
                }
    
    
                std::vector<double> costs(split_iters, 0.0);
                for (size_t i = 0; i < length; i++) {
                    double old_dist = dcache[start_idx + i];
                    for (size_t round = 0; round < split_iters; round++) {
                        double new_dist = flat_wpoints_l2_dot_distance<WithWeights>(data, squared_norms.data(), d, i, start_idx + 1 + round);
                        costs[round] += std::min(old_dist, new_dist);
                    }
                }
    
                // size_t best_index = std::numeric_limits<size_t>::max();
                double min_cost_center = std::numeric_limits<double>::max();
                for (size_t round = 0; round < split_iters; round++) {
                    if (costs[round] < min_cost_center) {
                        min_cost_center = costs[round];
                        best_index = random_pick_indices[round];
                    }
                }
            } else { // switch to indexes (which are relative to the current data span)
                
                { // Initializing random picks
                    size_t p_index = 0;
                    double cumulative = 0.0;
                    for (size_t i = 0; i < length; i++)
                    {
                        size_t rel_idx = indices[start_idx + i];
                        passert(rel_idx != std::numeric_limits<size_t>::max(), "Index should have been initialized, got {} at start_idx {}, length {}, i {}", rel_idx, start_idx, length, i);

                        cumulative += dcache[start_idx];

                        if (cumulative >= random_pick_probs[p_index])
                        {
                            random_pick_indices[p_index] = rel_idx;
                            p_index++;
                            if (p_index >= split_iters) break;
                        }
                    }
    
                    // ensure all picks are filled
                    if (p_index < split_iters) {
                        std::uniform_int_distribution<size_t> rand_index(1, length - 1);
                        for (; p_index < split_iters; p_index++) {
                            size_t rp = rand_index(rand_eng) + start_idx;
                            size_t rel_rp = indices[rp];
                            passert(rel_rp != std::numeric_limits<size_t>::max(), "Index should have been initialized, got {} at start_idx {}, length {}, rp {}", rel_rp, start_idx, length, rp);

                            random_pick_indices[p_index] = rel_rp;
                        }
                    }
                }

   
                double min_cost_center = std::numeric_limits<double>::max();    
    
                for (size_t round = 0; round < split_iters; round++)
                {
                    size_t random_point = random_pick_indices[round];
                    double cost = 0.0;
                    for (size_t i = 0; i < length; i++) {
                        size_t rel_idx = indices[start_idx + i];
                        passert(rel_idx != std::numeric_limits<size_t>::max(), "Index should have been initialized, got {} at start_idx {}, length {}, i {}", rel_idx, start_idx, length, i);

                        double old_dist = dcache[rel_idx];  
                        double new_dist = flat_wpoints_l2_dot_distance<WithWeights>(data, squared_norms.data(), d, rel_idx, random_point);
                        cost += std::min(old_dist, new_dist);
                    }
    
                    if (cost < min_cost_center)
                    {
                        min_cost_center = cost;
                        best_index = random_point;
                    }
                }
            }


            passert(best_index != std::numeric_limits<size_t>::max(), "Best index should have been set, got {} at iteration {}", best_index, choosen_k);
            return best_index;
        };

        auto partition_cluster = [&] (size_t start_idx, size_t length, size_t best_index) {
            passert(dcache[start_idx] == 0.0, "Current center should have zero distance to itself, got {}, start_idx {}, length {}", dcache[start_idx], start_idx, length);
            
            double right_cost = 0.0, left_cost = 0.0;
            size_t left_count = 0;


            if (length > Points_Per_L1) {
                std::vector<bool> is_left_mask(length, false);
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

                left_count = left_pos;
                move_point(best_index, left_count + start_idx);

                
                // initialize indices()
                if (left_count <= Points_Per_L1) {
                    std::cout << "Initializing left indices, left_count " << left_count << std::endl;
                    for (size_t i = 0; i < left_count; i++) {
                        indices[start_idx + i] = start_idx + i;
                    }
                }

                if (length - left_count <= Points_Per_L1) {

                    std::cout << "Initializing right indices, right_count " << (length - left_count) << std::endl;
                    for (size_t i = 0; i < length - left_count; i++) {
                        indices[start_idx + left_count + i] = start_idx + left_count + i;
                    }
                }
    
            } else {    

                std::cout << "Using index based partitioning, length " << length << std::endl;

                std::vector<size_t> left_indices;
                std::vector<size_t> right_indices;
                left_indices.push_back(start_idx); // center always goes to the left

                for (size_t i = 1; i < length; i++)
                {
                    size_t rel_idx = indices[start_idx + i];
                    double dist_old_center = dcache[rel_idx];
                    double dist_new_center = flat_wpoints_l2_dot_distance<WithWeights>(data, squared_norms.data(), d, rel_idx, best_index);

                    if (dist_old_center <= dist_new_center) {
                        left_cost += dist_old_center;
                        left_indices.push_back(rel_idx);
                    } else {
                        right_cost += dist_new_center;
                        dcache[rel_idx] = dist_new_center; // Update dcache for the right child
                        right_indices.push_back(rel_idx);
                    }
                }

                // update indices
                passert(!left_indices.empty(), "Left indices should not be empty, start_idx {}\
                    length {} best_index {}, rel_best_index {}",
                                            start_idx, length, best_index, best_index - start_idx);

                passert(!right_indices.empty(), "Right indices should not be empty, start_idx {}\
                    length {} best_index {}, rel_best_index {}",
                                            start_idx, length, best_index, best_index - start_idx);

                left_count = left_indices.size();
                for (size_t i = 0; i < left_indices.size(); i++) {
                    indices[start_idx + i] = left_indices[i];
                }

                for (size_t i = 0; i < right_indices.size(); i++) {
                    indices[start_idx + left_count + i] = right_indices[i];
                }

                // find new center in right cluster and move it to the start of the right cluster
                
                size_t found_in_right = std::numeric_limits<size_t>::max();
                for (size_t i = 0; i < right_indices.size(); i++) {
                    if (right_indices[i] == best_index) {
                        found_in_right = i;
                    }
                }

                std::cout << "found_in_right=" << found_in_right << " out of " << right_indices.size() << std::endl;
                std::cout << "best_index=" << best_index << " left_count=" << left_count << " start_idx=" << start_idx << std::endl;
                std::cout << "right_indices[found_in_right]=" << right_indices[found_in_right] << std::endl;
                std::cout << "indices[start_idx + left_count + found_in_right]=" << indices[start_idx + left_count + found_in_right] << std::endl;
                std::cout << "dcache[right_indices[found_in_right]]=" << dcache[right_indices[found_in_right]] << std::endl;    

                passert(found_in_right != std::numeric_limits<size_t>::max(), "New center should be in the right cluster, start_idx {} length {} best_index {}, rel_best_index {}",
                        start_idx, length, best_index, best_index - start_idx);

                passert(dcache[right_indices[found_in_right]] == 0.0, "New center should have zero distance to itself, got {}, start_idx {}, length {}, found_in_right {}, best_index {}, rel_best_index {}",
                        dcache[right_indices[found_in_right]], start_idx, length, found_in_right, best_index, best_index - start_idx);


                // Find wich cluster has left_count + start_idx
                // and swap it with the found_in_right
                // The first point of right cluster must match start_idx + left_count INDEX => indices[start_idx + left_count] = start_idx + left_count
                // To do so we need to find wich is the current point that matches start_idx + left_count
                // It can be in either the left or right cluster. If in the right cluster, 
                    // we just need to phisically swap it with best_index and update indices[start_idx + left_count] = start_idx + left_count and best_index = start_idx + left_count 
                    // and indices[best_index] = _SWAPPED_INDEX_
                    
                bool swapped = false;
                for (size_t i = 0; i < length; i++) {
                    if (indices[start_idx + i] == left_count + start_idx) {
                            std::swap(indices[start_idx + i], indices[best_index]);
                            move_point(best_index, start_idx + left_count);

                        swapped = true;
                        break;
                    }
                }

                passert(swapped, "Could not find left_count + start_idx in either cluster, left_count {} start_idx {} length {}",
                        left_count, start_idx, length);




                move_point(start_idx + left_count + found_in_right, left_count + start_idx);    
                std::swap(indices[start_idx + left_count + found_in_right], indices[left_count + start_idx]);

                passert(indices[start_idx] == start_idx, "Center should be at the start of the left cluster, got {} at start_idx {}", indices[start_idx], start_idx);
                passert(indices[start_idx + left_count] == best_index, "New center should be at the start of the right cluster, got {} at start_idx {}", indices[start_idx + left_count], start_idx + left_count);

                passert(dcache[indices[start_idx]] == 0.0, "Current center should have zero distance to itself, got {}, start_idx {}, length {}", dcache[indices[start_idx]], start_idx, length);
                passert(dcache[indices[start_idx + left_count]] == 0.0, "New center should have zero distance to itself, got {}, start_idx {}, length {}, left_count {}, best_index {}", dcache[indices[start_idx + left_count]], start_idx, length, left_count, best_index);

            }


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

            if (cluster_ref->props.split_count < 17) {

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

        std::cout << std::format("Finished splitting after {} iterations, got {} clusters / {} requested, percent {}%\n"
            , iters, choosen_k, k, static_cast<double>(choosen_k) / k * 100.0);

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
