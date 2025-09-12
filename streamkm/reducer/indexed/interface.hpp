#pragma once

#include "streamkm/reducer/indexed/cached.hpp"
#include "streamkm/reducer/indexed/naive.hpp"
#include "streamkm/reducer/indexed/sqnorms.hpp"
#include "streamkm/reducer/indexed/sqnorms_cached.hpp"

namespace streamkm
{
    
    using IndexTCoresetReducer = _IndexCoresetReducer<CoresetReducerNoMetrics, TreeCoresetClusterSet>;
    using IndexFCoresetReducer = _IndexCoresetReducer<CoresetReducerNoMetrics, FenwickCoresetClusterSet>;
    using IndexTCachedCoresetReducer = _IndexCachedCoresetReducer<CoresetReducerNoMetrics, TreeCoresetClusterSet>;
    using IndexFCachedCoresetReducer = _IndexCachedCoresetReducer<CoresetReducerNoMetrics, FenwickCoresetClusterSet>;
    using IndexTSqNormsCoresetReducer = _IndexSqNormsCoresetReducer<CoresetReducerNoMetrics, TreeCoresetClusterSet>;
    using IndexFSqNormsCoresetReducer = _IndexSqNormsCoresetReducer<CoresetReducerNoMetrics, FenwickCoresetClusterSet>;
    using IndexTSqNormsCachedCoresetReducer = _IndexSqNormsCachedCoresetReducer<CoresetReducerNoMetrics, TreeCoresetClusterSet>;
    using IndexFSqNormsCachedCoresetReducer = _IndexSqNormsCachedCoresetReducer<CoresetReducerNoMetrics, FenwickCoresetClusterSet>;

    using IndexTCoresetReducerWMetrics = _IndexCoresetReducer<CoresetReducerChronoMetrics, TreeCoresetClusterSet>;
    using IndexFCoresetReducerWMetrics = _IndexCoresetReducer<CoresetReducerChronoMetrics, FenwickCoresetClusterSet>;
    using IndexTCachedCoresetReducerWMetrics = _IndexCachedCoresetReducer<CoresetReducerChronoMetrics, TreeCoresetClusterSet>;
    using IndexFCachedCoresetReducerWMetrics = _IndexCachedCoresetReducer<CoresetReducerChronoMetrics, FenwickCoresetClusterSet>;
    using IndexTSqNormsCoresetReducerWMetrics = _IndexSqNormsCoresetReducer<CoresetReducerChronoMetrics, TreeCoresetClusterSet>;
    using IndexFSqNormsCoresetReducerWMetrics = _IndexSqNormsCoresetReducer<CoresetReducerChronoMetrics, FenwickCoresetClusterSet>;
    using IndexTSqNormsCachedCoresetReducerWMetrics = _IndexSqNormsCachedCoresetReducer<CoresetReducerChronoMetrics, TreeCoresetClusterSet>;
    using IndexFSqNormsCachedCoresetReducerWMetrics = _IndexSqNormsCachedCoresetReducer<CoresetReducerChronoMetrics, FenwickCoresetClusterSet>;

    // _IndexSqNormsCachedCoresetReducer

} // namespace streamkm
