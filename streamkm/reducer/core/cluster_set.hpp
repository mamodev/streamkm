#pragma once

#include "streamkm/core/errors.hpp"
#include "streamkm/core/traits.hpp"
#include "streamkm/core/random.hpp"
#include "streamkm/core/fenwick.hpp"


#include <concepts>
#include <random>
#include <stack>

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
concept _CCS_API = requires(T a, typename T::TCluster& c, double p_cost, double c_cost, typename T::TCluster::TProps p_props, typename T::TCluster::TProps c_props) {
    { a.total_cost() } -> std::same_as<double>;
    { a.pick() } -> std::same_as<typename T::TCluster&>;
    { a.cluster_split(c, p_cost, c_cost, std::move(p_props), std::move(c_props)) }
        -> std::same_as<void>;};

template <typename T>
concept _CCS_TCluster = requires {
    typename T::TCluster;
    requires _TCluster<typename T::TCluster>;
};

template <typename T>
concept CoresetClusterSet = _CCS_API<T> && _CCS_TCluster<T>;


template <template <MovableNotCopyable> typename CCS>
concept CoresetClusterSetFamily = requires {
    // Must work when instantiated on some Props
    typename CCS<__dummy_tcluster>::TCluster;
    requires _CCS_API<CCS<__dummy_tcluster>>;
    requires _CCS_TCluster<CCS<__dummy_tcluster>>;
};

template <typename T>
class LeafIteratorT {
private:
    // underlying we still track pointers,
    // but produce references on deref
    std::stack<T*> st;

    void advance_to_leaf() {
        while (!st.empty()) {
            T* cur = st.top();
            if (cur->is_leaf())
                return; // found a leaf
            st.pop();
            if (cur->rc) st.push(cur->rc);
            if (cur->lc) st.push(cur->lc);
        }
    }

public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = T;
    using difference_type   = std::ptrdiff_t;
    using pointer           = T*;
    using reference         = T&;

    LeafIteratorT() = default;
    explicit LeafIteratorT(T* root) {
        if (root) st.push(root);
        advance_to_leaf();
    }

    reference operator*() const { return *st.top(); }
    pointer operator->() const { return st.top(); }

    // ++it
    LeafIteratorT& operator++() {
        st.pop();
        advance_to_leaf();
        return *this;
    }

    // it++
    LeafIteratorT operator++(int) {
        LeafIteratorT tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const LeafIteratorT& other) const {
        if (st.empty() && other.st.empty()) return true;
        if (st.empty() || other.st.empty()) return false;
        return st.top() == other.st.top();
    }

    bool operator!=(const LeafIteratorT& other) const {
        return !(*this == other);
    }
};

template <MovableNotCopyable Props>
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


    TreeCoresetClusterSet(size_t nclusters, std::pair<Props, double>&& init_root,  URBG& rand_eng) :
        arena(nclusters * 2 + 1),
        rand_eng(rand_eng),
        rand_double(0.0, 1.0) {

        root = arena.alloc();
        *root = Node{
            .parent = nullptr,
            .lc = nullptr,
            .rc = nullptr,
            ._cost = init_root.second,
            .props = std::move(init_root.first)
        };
    }

    double total_cost() const {
        return root->_cost;
    }

    Node& pick() {
        double prob = rand_double(rand_eng);
        Node *node = root;
        
        while (!node->is_leaf())
            if (prob < node->lc->_cost / node->_cost)
                node = node->lc;
            else
                node = node->rc;

        return std::ref(*node);
    }

    void cluster_split(Node& origin, double new_parent_cost, double new_cluster_cost, Props&& origin_props, Props&& new_cluster_props) {

        passert(origin.is_leaf(), "Can only split leaf nodes is_leaf={}", origin.is_leaf());
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

        origin.props = Props();

        Node* curr_node = origin.parent;
        while(curr_node) {
            curr_node->_cost = curr_node->lc->_cost + curr_node->rc->_cost;
            curr_node = curr_node->parent;
        }
    }


    LeafIteratorT<Node> begin() {
        return LeafIteratorT<Node>(root);
    }
    LeafIteratorT<Node> end() {
        return LeafIteratorT<Node>();
    }
    LeafIteratorT<const Node> begin() const {
        return LeafIteratorT<const Node>(root);
    }
    LeafIteratorT<const Node> end() const {
        return LeafIteratorT<const Node>();
    }

private:
    URBG& rand_eng;
    NodeArena arena;
    Node* root = nullptr;
    std::uniform_real_distribution<double> rand_double;
};

static_assert(CoresetClusterSetFamily<TreeCoresetClusterSet>, "TreeCoresetClusterSet is not a CoresetClusterSetFamily");


template <MovableNotCopyable Props>
class FenwickCoresetClusterSet  {

public:

struct Cluster {
    using TProps = Props;
    size_t index = 0;
    double _cost = 0.0;
    Props props;
    inline double cost() const { return _cost; }
};

using TCluster = Cluster;


FenwickCoresetClusterSet(size_t nclusters, std::pair<Props, double>&& init_root, URBG& rand_eng) :
    _m_cluster_costs(nclusters),
    _m_clusters(),
    _m_total_cost(init_root.second),
    _m_rand_eng(rand_eng),
    _m_rand_double(0.0, 1.0) {

    _m_clusters.reserve(nclusters);
    _m_clusters.push_back(Cluster{
        .index = 0,
        ._cost = init_root.second,
        .props = std::move(init_root.first)
    });

    _m_cluster_costs.update(1, init_root.second);
}

double total_cost() const noexcept {
    return _m_total_cost;
}

Cluster& pick() {
    passert(!_m_clusters.empty(), "No clusters to pick from", "");

    size_t idx = _m_cluster_costs.bin_search_ge(_m_rand_double(_m_rand_eng) * total_cost()) - 1;
    passert(idx < _m_clusters.size(), "Cluster index should be valid, got {} >= {}", idx, _m_clusters.size());

    return _m_clusters[idx];
}


void cluster_split(Cluster& origin, double new_parent_cost, double new_cluster_cost, Props&& origin_props, Props&& new_cluster_props) {
    passert(new_parent_cost >= 0.0, "Parent cost should be non-negative, got {}", new_parent_cost);
    passert(new_cluster_cost >= 0.0, "New cluster cost should be non-negative, got {}", new_cluster_cost);
    
    size_t index = _m_clusters.size();

    _m_cluster_costs.update(index + 1, new_cluster_cost);
    _m_cluster_costs.update(origin.index + 1, new_parent_cost - origin._cost);
    _m_total_cost += (new_parent_cost + new_cluster_cost) - origin._cost;

    origin._cost = new_parent_cost;
    origin.props = std::move(origin_props);

    _m_clusters.push_back(Cluster{
        .index = index,
        ._cost = new_cluster_cost,
        .props = std::move(new_cluster_props)
    });
}

    auto begin() { return _m_clusters.begin(); }
    auto end() { return _m_clusters.end(); }
    auto begin() const { return _m_clusters.begin(); }
    auto end() const { return _m_clusters.end(); }

private:
    double _m_total_cost;
    FenwickTree<double, size_t> _m_cluster_costs;
    std::vector<Cluster> _m_clusters;
    URBG& _m_rand_eng;
    std::uniform_real_distribution<double> _m_rand_double;
};

static_assert(CoresetClusterSetFamily<FenwickCoresetClusterSet>, "FenwickCoresetClusterKWSet is not a CoresetClusterSetFamily");

} // namespace streamkm