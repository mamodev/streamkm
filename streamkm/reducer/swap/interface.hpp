#pragma once 

#include "streamkm/reducer/swap/naive.hpp"

namespace streamkm {
    using SwapTCoresetReducer = _SwapCoresetReducer<CoresetReducerNoMetrics, TreeCoresetClusterSet>;
    using SwapFCoresetReducer = _SwapCoresetReducer<CoresetReducerNoMetrics, FenwickCoresetClusterSet>;

    using SwapTCoresetReducerWMetrics = _SwapCoresetReducer<CoresetReducerChronoMetrics, TreeCoresetClusterSet>;
    using SwapFCoresetReducerWMetrics = _SwapCoresetReducer<CoresetReducerChronoMetrics, FenwickCoresetClusterSet>;
}