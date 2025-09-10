#pragma once
#include <vector>
#include <random>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <limits>

// -----------------------------------------------------------------------------
// KMeans++ initialization
// -----------------------------------------------------------------------------
static inline std::vector<float> kmeanspp(
    const float* data, std::size_t N, std::size_t D,
    std::size_t K_in, std::mt19937_64& rng) {

  if (N == 0 || D == 0 || K_in == 0) return {};
  const std::size_t K = std::min(K_in, N);

  std::vector<float> centers(K * D);
  std::vector<double> best_dist2(N, std::numeric_limits<double>::max());

  // choose first center uniformly at random
  std::uniform_int_distribution<std::size_t> uid(0, N - 1);
  std::size_t first_idx = uid(rng);
  for (std::size_t d = 0; d < D; ++d)
    centers[d] = data[first_idx * D + d];

  auto sq_dist_point = [&](std::size_t i, const float* center) {
    double s = 0.0;
    const float* pi = data + i * D;
    for (std::size_t d = 0; d < D; ++d) {
      double diff = double(pi[d]) - double(center[d]);
      s += diff * diff;
    }
    return s;
  };

  // initialize distances
  for (std::size_t i = 0; i < N; ++i)
    best_dist2[i] = sq_dist_point(i, centers.data());

  // choose remaining centers
  for (std::size_t c = 1; c < K; ++c) {
    double sum = std::accumulate(best_dist2.begin(), best_dist2.end(), 0.0);

    std::size_t next_idx = 0;
    if (sum <= 0.0) {
      next_idx = uid(rng);
    } else {
      std::uniform_real_distribution<double> unif(0.0, sum);
      double r = unif(rng), cum = 0.0;
      for (std::size_t i = 0; i < N; ++i) {
        cum += best_dist2[i];
        if (r <= cum) { next_idx = i; break; }
      }
    }

    // add new center
    float* center_ptr = &centers[c * D];
    const float* src = data + next_idx * D;
    for (std::size_t d = 0; d < D; ++d)
      center_ptr[d] = src[d];

    // update distances
    for (std::size_t i = 0; i < N; ++i) {
      double d2 = sq_dist_point(i, center_ptr);
      if (d2 < best_dist2[i]) best_dist2[i] = d2;
    }
  }

  return centers;
}

// -----------------------------------------------------------------------------
// Assignment step (just labels, no distances)
// -----------------------------------------------------------------------------
inline std::vector<std::size_t> assign_labels(const std::size_t D, const std::vector<float>& centers, const float* data, const std::size_t data_size) {
  const std::size_t K = centers.size() / D;
  const std::size_t N = data_size / D;
  std::vector<std::size_t> labels(N, std::numeric_limits<std::size_t>::max());

  for (std::size_t i = 0; i < N; ++i) {
    const float* pi = data + i * D;
    double best_dist = std::numeric_limits<double>::max();
    std::size_t best_k = 0;
    
    for (std::size_t k = 0; k < K; ++k) {
      const float* ck = &centers[k * D];
      double dist = 0.0;
      
      for (std::size_t d = 0; d < D; ++d) {
        double diff = double(pi[d]) - double(ck[d]);
        dist += diff * diff;
      }

      if (dist < best_dist) {
        best_dist = dist;
        best_k = k;
      }

    }
    
    labels[i] = best_k;
  }

  return labels;
}

// -----------------------------------------------------------------------------
// Full KMeans (Lloyd's algorithm)
// -----------------------------------------------------------------------------
inline std::vector<float> kmeans(
    const float* data, std::size_t N, std::size_t D,
    std::size_t K, std::size_t max_iters = 100,
    double tol = 1e-4, unsigned int seed = 42) {

  std::mt19937_64 rng(seed);
  std::vector<float> centers = kmeanspp(data, N, D, K, rng);
  std::vector<float> new_centers(K * D, 0.0f);
  std::vector<std::size_t> counts(K);

  for (std::size_t it = 0; it < max_iters; ++it) {
    // auto labels = assign_labels(centers, data, N, D);
    auto labels = assign_labels(D, centers, data, N * D);

    // recompute means
    std::fill(new_centers.begin(), new_centers.end(), 0.0f);
    std::fill(counts.begin(), counts.end(), 0);

    for (std::size_t i = 0; i < N; ++i) {
      std::size_t lbl = labels[i];
      const float* p = data + i * D;
      float* c = &new_centers[lbl * D];
      for (std::size_t d = 0; d < D; ++d) c[d] += p[d];
      counts[lbl]++;
    }

    for (std::size_t k = 0; k < K; ++k) {
      float* c = &new_centers[k * D];
      if (counts[k] > 0) {
        for (std::size_t d = 0; d < D; ++d) c[d] /= counts[k];
      } else {
        // empty cluster reinit
        std::uniform_int_distribution<std::size_t> uid(0, N - 1);
        const float* p = data + uid(rng) * D;
        for (std::size_t d = 0; d < D; ++d) c[d] = p[d];
      }
    }

    // check convergence
    double diff = 0.0;
    for (std::size_t i = 0; i < K * D; ++i) {
      double delta = double(new_centers[i]) - double(centers[i]);
      diff += delta * delta;
    }
    centers.swap(new_centers);
    if (diff < tol) break;
  }

  return centers;
}