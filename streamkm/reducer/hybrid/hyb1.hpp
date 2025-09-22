
#include "streamkm/core/random.hpp"
#include "streamkm/core/traits.hpp"

#include "streamkm/reducer/core/all.hpp"
#include "streamkm/reducer/core/reducer.hpp"

#include <span>
#include <vector>


namespace streamkm
{
struct __SqNormsCachedSwapClusterProps {
    std::span<float>    data;
    std::span<double>   sqnorms; // cached squared norms for each point
    std::span<double>   dcache; // cached distance to current center

    __SqNormsCachedSwapClusterProps(std::span<float> d, std::span<double> sn, std::span<double> dc) 
        : data(d), sqnorms(sn), dcache(dc) {}
        
     __SqNormsCachedSwapClusterProps() = default;
    // Make it movable but not copyable
    __SqNormsCachedSwapClusterProps(const __SqNormsCachedSwapClusterProps&) = delete;
    __SqNormsCachedSwapClusterProps(__SqNormsCachedSwapClusterProps&&) = default;
    __SqNormsCachedSwapClusterProps& operator=(const __SqNormsCachedSwapClusterProps&) = delete;
    __SqNormsCachedSwapClusterProps& operator=(__SqNormsCachedSwapClusterProps&&) = default;

};


template <typename Metrics, template <MovableNotCopyable> typename CCS>
requires CoresetClusterSetFamily<CCS>
class _Hyb1CoresetReducer {
public:
    using Result = FlatWPointsResult;
    Metrics::MetricsSet metrics;

    template <bool WithWeights>
    inline Result __reduce(float *const data, const std::size_t n, const std::size_t d, const std::size_t k)
   {
        passert(n < std::numeric_limits<size_t>::max(), "Too many points, cannot handle more than {} points", std::numeric_limits<size_t>::max());

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

static_assert(
    CoresetReducer<_Hyb1CoresetReducer<CoresetReducerNoMetrics, FenwickCoresetClusterSet>>, 
    "_Hyb1CoresetReducer does not satisfy CoresetReducer concept"
);


}