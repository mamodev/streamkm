#pragma once 

#include <vector>

template<typename T, typename TIndex = int>
struct FenwickTree {
    TIndex n;
    std::vector<T> bit; // store values of type T

    FenwickTree(TIndex n) {
        this->n = n;
        bit.assign(n + 1, T());
    }

    // add delta to position idx
    void update(TIndex idx, T delta) {
        for (; idx <= n; idx += idx & -idx)
            bit[idx] += delta;
    }

    // prefix sum [1..idx]
    T query(TIndex idx) {
        T sum = 0.0;
        for (; idx > 0; idx -= idx & -idx)
            sum += bit[idx];
        return sum;
    }

    // range sum [l..r]
    T rangeQuery(TIndex l, TIndex r) {
        return query(r) - query(l - 1);
    }

    // returns the smallest index in [1..n] such that prefix sum >= x.
    // If x <= 0 returns 1. If x > total sum returns n + 1 as "not found".
    TIndex bin_search_ge(T x) {
        if (x <= T()) return 1;               // handle non-positive x
        T total = query(n);
        if (x > total) return n + 1;          // not found sentinel

        TIndex idx = 0;
        T curr = T();

        // largest power of two <= n
        TIndex bitMask = 1;
        while ((bitMask << 1) <= n) bitMask <<= 1;

        for (; bitMask != 0; bitMask >>= 1) {
            TIndex next = idx + bitMask;
            if (next <= n && curr + bit[next] < x) {
                idx = next;
                curr += bit[next];
            }
        }
        return idx + 1;
    }
    
};
