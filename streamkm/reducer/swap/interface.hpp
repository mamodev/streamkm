#pragma once 

#include "streamkm/reducer/swap/naive.hpp"
#include "streamkm/reducer/swap/sqnorms_cached.hpp"

namespace streamkm {
    using SwapTCoresetReducer = _SwapCoresetReducer<CoresetReducerNoMetrics, TreeCoresetClusterSet>;
    using SwapFCoresetReducer = _SwapCoresetReducer<CoresetReducerNoMetrics, FenwickCoresetClusterSet>;

    using SwapTSqNormsCachedCoresetReducer = _SwapSqNormsCachedCoresetReducer<CoresetReducerNoMetrics, TreeCoresetClusterSet>;
    using SwapFSqNormsCachedCoresetReducer = _SwapSqNormsCachedCoresetReducer<CoresetReducerNoMetrics, FenwickCoresetClusterSet>;

    using SwapTCoresetReducerWMetrics = _SwapCoresetReducer<CoresetReducerChronoMetrics, TreeCoresetClusterSet>;
    using SwapFCoresetReducerWMetrics = _SwapCoresetReducer<CoresetReducerChronoMetrics, FenwickCoresetClusterSet>;
    using SwapTSqNormsCachedCoresetReducerWMetrics = _SwapSqNormsCachedCoresetReducer<CoresetReducerChronoMetrics, TreeCoresetClusterSet>;
    using SwapFSqNormsCachedCoresetReducerWMetrics = _SwapSqNormsCachedCoresetReducer<CoresetReducerChronoMetrics, FenwickCoresetClusterSet>;
}