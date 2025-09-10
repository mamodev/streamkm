#pragma once

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

class ContingencyMatrix {
    std::size_t k_true;
    std::size_t k_pred;

    std::vector<std::size_t> contingency;

    std::vector<std::size_t> true_cluster_sizes;      // a[i]
    std::vector<std::size_t> predicted_cluster_sizes; // b[j]
    std::size_t total;                                // n

    static inline double comb2(std::size_t x) {
        return x < 2 ? 0.0 : (double)x * (x - 1) / 2.0;
    }

public:
    ContingencyMatrix(std::size_t k_true_,
                      std::size_t k_pred_,
                      const std::vector<std::size_t>& true_clusters,
                      const std::vector<std::size_t>& pred_clusters)
        : k_true(k_true_), k_pred(k_pred_),
          contingency(k_true_ * k_pred_, 0),
          true_cluster_sizes(k_true_, 0),
          predicted_cluster_sizes(k_pred_, 0),
          total(true_clusters.size()) {

        for (std::size_t idx = 0; idx < true_clusters.size(); idx++) {
            std::size_t ti = true_clusters[idx];
            std::size_t pj = pred_clusters[idx];
            contingency[ti * k_pred + pj]++;
            true_cluster_sizes[ti]++;
            predicted_cluster_sizes[pj]++;
        }
    }

    inline std::size_t n(std::size_t i, std::size_t j) const {
        return contingency[i * k_pred + j];
    }

    inline std::size_t a(std::size_t i) const { return true_cluster_sizes[i]; }
    inline std::size_t b(std::size_t j) const { return predicted_cluster_sizes[j]; }
    inline std::size_t N() const { return total; }

    std::size_t true_k() const { return k_true; }
    std::size_t pred_k() const { return k_pred; }

    // ------------------------- Metrics ------------------------

    // Purity: average max-overlap per predicted cluster
    double purity() const {
        double sum = 0;
        for (std::size_t j = 0; j < k_pred; j++) {
            std::size_t max_overlap = 0;
            for (std::size_t i = 0; i < k_true; i++) {
                max_overlap = std::max(max_overlap, n(i, j));
            }
            sum += max_overlap;
        }
        return sum / N();
    }

    // Adjusted Rand Index
    double adjusted_rand_index() const {
        double sum_index = 0;
        for (std::size_t i = 0; i < k_true; i++) {
            for (std::size_t j = 0; j < k_pred; j++) {
                sum_index += comb2(n(i, j));
            }
        }

        double sum_a = 0;
        for (std::size_t i = 0; i < k_true; i++) sum_a += comb2(a(i));
        double sum_b = 0;
        for (std::size_t j = 0; j < k_pred; j++) sum_b += comb2(b(j));

        double nC2 = comb2(N());
        double expected_index = (sum_a * sum_b) / nC2;
        double max_index = 0.5 * (sum_a + sum_b);

        return (sum_index - expected_index) / (max_index - expected_index);
    }

    // Normalized Mutual Information
    double normalized_mutual_info() const {
        double Nf = static_cast<double>(N());
        double mi = 0.0;

        for (std::size_t i = 0; i < k_true; i++) {
            for (std::size_t j = 0; j < k_pred; j++) {
                double nij = n(i, j);
                if (nij == 0) continue;
                mi += (nij / Nf) *
                      std::log((nij * Nf) /
                               (static_cast<double>(a(i)) * b(j)));
            }
        }

        double hu = 0.0;
        for (std::size_t i = 0; i < k_true; i++) {
            double ai = a(i);
            if (ai > 0) hu -= (ai / Nf) * std::log(ai / Nf);
        }

        double hv = 0.0;
        for (std::size_t j = 0; j < k_pred; j++) {
            double bj = b(j);
            if (bj > 0) hv -= (bj / Nf) * std::log(bj / Nf);
        }

        return (hu == 0 || hv == 0) ? 0 : mi / std::sqrt(hu * hv);
    }

    // Fowlkes–Mallows Index
    double fowlkes_mallows() const {
        double tp = 0.0; // true positive pair count
        for (std::size_t i = 0; i < k_true; i++) {
            for (std::size_t j = 0; j < k_pred; j++) {
                tp += comb2(n(i, j));
            }
        }

        double fp = 0.0; // false positives
        for (std::size_t j = 0; j < k_pred; j++) {
            fp += comb2(b(j));
        }
        fp -= tp;

        double fn = 0.0; // false negatives
        for (std::size_t i = 0; i < k_true; i++) {
            fn += comb2(a(i));
        }
        fn -= tp;

        return tp == 0 ? 0.0 : tp / std::sqrt((tp + fp) * (tp + fn));
    }
};