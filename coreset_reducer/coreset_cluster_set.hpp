#pragma once

#include "core/all.hpp"   

#include <concepts>
#include <random>


namespace streamkm
{

template <typename T>
concept _TCluster = requires(const T c, T x) {
    { c.cost() } -> std::same_as<double>;
    { x.props } -> std::same_as<typename std::remove_cvref_t<T>::TProps&>;
}  && MovableNotCopyable<T>;

struct __dummy_tcluster {
    struct _Props {
        _Props(const _Props&) = delete;
        _Props(_Props&&) = default;
        _Props& operator=(_Props&&) = default;
        int dummy = 0;
    };

    using TProps = _Props;

    double cost() const { return 0.0; }
    _Props props;
};

static_assert(_TCluster<__dummy_tcluster>, "__dummy_tcluster does not satisfy _TCluster concept");

template <typename T>
concept _CCS_API = requires(T a, typename T::TCluster& c, double p_cost, double c_cost, typename T::TCluster::TProps& p_props, typename T::TCluster::TProps& c_props) {
    { a.total_cost() } -> std::same_as<double>;
    { a.pick() } -> std::same_as<std::reference_wrapper<typename T::TCluster>>;
    { a.cluster_split(c, p_cost, c_cost, p_props, c_props) } -> std::same_as<void>;
};

template <typename T>
concept _CCS_TCluster = requires {
    typename T::TCluster;
    requires _TCluster<typename T::TCluster>;
};

template <typename T>
concept CoresetClusterSet = _CCS_API<T> && _CCS_TCluster<T>;

template <MovableNotCopyable Props, std::uniform_random_bit_generator RandEng>
class TreeCoresetClusterSet {

public:
    struct Node
    {
        using TProps = Props;
        Node *parent = nullptr;
        Node *lc = nullptr;
        Node *rc = nullptr;
        double _cost = 0.0;
        Props props;

        inline bool is_leaf() const { return lc == nullptr && rc == nullptr; }
        inline double cost() const { return _cost; }
    };
    using TCluster = Node;

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

    TreeCoresetClusterSet(size_t nclusters, Props& root_props, double root_cost, RandEng& rand_eng) :
        arena(nclusters * 2 + 1),
        rand_eng(rand_eng),
        rand_double(0.0, 1.0) {

        root = arena.alloc();
        *root = Node{
            .parent = nullptr,
            .lc = nullptr,
            .rc = nullptr,
            ._cost = root_cost,
            .props = std::move(root_props)
        };
    }

    double total_cost() const {
        return root->_cost;
    }

    std::reference_wrapper<Node> pick() {
        double prob = rand_double(rand_eng);
        Node *node = root;
        
        while (!node->is_leaf())
            if (prob < node->lc->_cost / node->_cost)
                node = node->lc;
            else
                node = node->rc;

        return std::ref(*node);
    }

    void cluster_split(Node& origin, double new_parent_cost, double new_cluster_cost, Props& origin_props, Props& new_cluster_props) {

        passert(origin.is_leaf(), "Can only split leaf nodes");
        passert(new_parent_cost >= 0.0, "Parent cost should be non-negative, got {}", new_parent_cost);
        passert(new_cluster_cost >= 0.0, "New cluster cost should be non-negative, got {}", new_cluster_cost);

        Node *lc = arena.alloc();
        Node *rc = arena.alloc();

        *lc = Node{
            .parent = &origin,
            .lc = nullptr,
            .rc = nullptr,
            ._cost = new_parent_cost,
            .props = std::move(origin_props)
        };

        *rc = Node{
            .parent = &origin,
            .lc = nullptr,
            .rc = nullptr,
            ._cost = new_cluster_cost,
            .props = std::move(new_cluster_props)
        };

        origin.lc = lc;
        origin.rc = rc;
        origin._cost = new_parent_cost + new_cluster_cost;
        origin.props = Props{};
    }


private:
    RandEng& rand_eng;
    NodeArena arena;
    Node* root = nullptr;
    std::uniform_real_distribution<double> rand_double;
};

static_assert(CoresetClusterSet<TreeCoresetClusterSet<__dummy_tcluster, std::mt19937>>, "TreeCoresetClusterSet does not satisfy CoresetClusterSet concept");

template <MovableNotCopyable Props, std::uniform_random_bit_generator RandEng>
class FenwickCoresetClusterSet {

public:

using TCluster = Props;

struct Cluster {
    using TProps = Props;
    size_t index = 0;
    double _cost = 0.0;
    Props props;
    inline double cost() const { return _cost; }
};


FenwickCoresetClusterSet(size_t capacity, Props& root_props, double root_cost, RandEng& rand_eng) :
    cluster_costs(capacity),
    clusters(),
    total_cost(root_cost),
    rand_eng(rand_eng),
    rand_double(0.0, 1.0) {
    
    clusters.reserve(capacity);
    clusters.push_back(Cluster{
        .index = 0,
        ._cost = root_cost,
        .props = std::move(root_props)
    });

    cluster_costs.update(1, root_cost);
}

double total_cost() const {
    return total_cost;
}

std::reference_wrapper<Props> pick() {
    passert(!clusters.empty(), "No clusters to pick from");

    size_t idx = cluster_costs.bin_search_ge(rand_double(rand_eng) * total_cost) - 1;
    passert(idx < clusters.size(), "Cluster index should be valid, got {} >= {}", idx, clusters.size());

    return std::ref(clusters[idx]);
}

void cluster_split(Node& origin, double new_parent_cost, double new_cluster_cost, Props& origin_props, Props& new_cluster_props) {
    passert(new_parent_cost >= 0.0, "Parent cost should be non-negative, got {}", new_parent_cost);
    passert(new_cluster_cost >= 0.0, "New cluster cost should be non-negative, got {}", new_cluster_cost);

    cluster_costs.update(index + 1, new_cluster_cost);
    cluster_costs.update(origin.index + 1, new_parent_cost - origin._cost);
    total_cost += (new_parent_cost + new_cluster_cost) - origin._cost;

    origin._cost = new_parent_cost;
    origin.props = std::move(origin_props);

    size_t index = clusters.size();
    clusters.push_back(Cluster{
        .index = index,
        ._cost = new_cluster_cost,
        .props = std::move(new_cluster_props)
    });
}

private:
    double total_cost;
    FenwickTree<double, size_t> cluster_costs;
    std::vector<Cluster> clusters;

    RandEng& rand_eng;
    std::uniform_real_distribution<double> rand_double;
};


} // namespace streamkm