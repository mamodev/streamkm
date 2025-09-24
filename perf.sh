#!/bin/bash
set -e
TMP_DIR=$(mktemp -d)
FIFO_PREFIX="$TMP_DIR/fifo"

mkfifo "$FIFO_PREFIX.ack"
mkfifo "$FIFO_PREFIX.ctl"

exec {perf_ctl_fd}<>"$FIFO_PREFIX.ctl"
exec {perf_ack_fd}<>"$FIFO_PREFIX.ack"
export PERF_CTL_FD=${perf_ctl_fd}
export PERF_ACK_FD=${perf_ack_fd}


nmi_watchdog=$(cat /proc/sys/kernel/nmi_watchdog)
perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)

echo 0  | sudo tee /proc/sys/kernel/nmi_watchdog > /dev/null
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null

function cleanup {
    echo "Cleaning up..."
    echo "$nmi_watchdog" | sudo tee /proc/sys/kernel/nmi_watchdog > /dev/null
    echo "$perf_event_paranoid" | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null

    rm "$FIFO_PREFIX.ack"
    rm "$FIFO_PREFIX.ctl"
    rmdir "$TMP_DIR"
}

trap cleanup EXIT

# prefix with date
STAT_OUT_FILE="perf_stat_$(date +%Y%m%d_%H%M%S).log"

perf stat \
  -e L1-dcache-loads \
  -e L1-dcache-load-misses \
  -e L1-icache-loads \
  -e L1-icache-load-misses \
  -e l2_cache_accesses_from_dc_misses \
  -e l2_cache_accesses_from_ic_misses \
  -e l2_cache_hits_from_dc_misses \
  -e l2_cache_hits_from_ic_misses \
  -e l2_cache_misses_from_dc_misses \
  -e l2_cache_misses_from_ic_miss \
  -e cache-misses \
  -e cache-references \
  -e l1_dtlb_misses \
  -e l2_dtlb_misses \
  -e l2_itlb_misses \
  -e LLC-loads \
  -e LLC-load-misses \
  -e LLC-stores \
  -e LLC-store-misses \
  -r 5 \
  --control fd:$PERF_CTL_FD,$PERF_ACK_FD \
  -o "$STAT_OUT_FILE" \
  -- .build/main "$@"

echo "Performance statistics saved to $STAT_OUT_FILE"


# perf record \
#   -g \
#   -e L1-dcache-load-misses \
#   -e L1-icache-load-misses \
#   -e l2_cache_misses_from_dc_misses \
#   -e l2_cache_misses_from_ic_miss \
#   -e cache-misses \
#   -e l1_dtlb_misses \
#   -e l2_dtlb_misses \
#   -e l2_itlb_misses \
#   --control fd:$PERF_CTL_FD,$PERF_ACK_FD \
#   -- ./main "$@"