#! /bin/bash
set -e


# using Metrics = streamkm::CoresetReducerChronoMetrics;
# // using Metrics = streamkm::CoresetReducerNoMetrics;
# using RandEng = xorshift128plus;
# auto cres = streamkm::coreset_serial_stream<streamkm::SwapFenwickCoresetReducer<0U, RandEng, Metrics>>(stream);


# desired usage = auto cres = DYN_STREAM(true, true, "xorshift128plus", "SwapFenwickCoresetReducer", "serial")(stream);

printf "#define DYN_STREAM(deterministic, with_metrics, reducer_name, stream_type) ({ \\\n"


printf "})\n"

