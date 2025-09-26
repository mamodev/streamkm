set -e

# usage: ./build.sh {flags} -- {args to program}


RUN_CMD="no"

FLAGS=()
ARGS=()

for arg in "$@"; do
    if [ "$arg" == "--" ]; then
        RUN_CMD="yes"
        continue
    fi

    if [ "$RUN_CMD" == "no" ]; then
        FLAGS+=("$arg")
    else
        ARGS+=("$arg")
    fi
done


FORCE_REBUILD="no"
for f in "${FLAGS[@]}"; do
    if [ "$f" == "-f" ] || [ "$f" == "--force" ]; then
        FORCE_REBUILD="yes"
    fi
done


mkdir -p .build

CC=g++
CFLAGS="--std=c++20 -I. "
# CFLAGS+=" -O0 -g -fsanitize=address -fno-omit-frame-pointer"
# CFLAGS+=" -O0 -g  -fno-omit-frame-pointer"
CFLAGS+=" -O3 -ffast-math -march=native"
# CFLAGS+=" -O3 -ffast-math -march=native -DDISABLE_PASSERT"
# CFLAGS+=" -O3 -ffast-math -march=native -DDISABLE_PASSERT"

# CFLAGS+=" -g -O1 -march=native -fsanitize=address -fno-omit-frame-pointer"

# g++ -O3 -ffast-math -march=native --std=c++20 -I. -o main main.cpp        -DDISABLE_PASSERT

    # clang++ -O3 -ffast-math -march=native --std=c++20 -I. -o main main.cpp    -DDISABLE_PASSERT -lopenblas

    # clang++ -O2 -g -fno-omit-frame-pointer -ffast-math -march=native --std=c++20 -I. -o main main.cpp -DDISABLE_PASSERT

HEADERS_FILES=$(g++ -E -M -I. ./cmd/main.cpp)

FILTER_H_W_PREFIXES=(
    /usr/include
    /usr/local/include
    /usr/lib/
    /usr/local/lib/
)

SOURCE_DEPS=("cmd/main.cpp")

for h in $HEADERS_FILES; do
    valid=true
    for p in "${FILTER_H_W_PREFIXES[@]}"; do
        if [[ $h == $p* || $h == "\\" ]]; then
        valid=false
        break
        fi

    done

    if [ "$valid" = false ]; then
        continue
    fi

    # replace any ./  with nothing
    h=${h//.\/}
    SOURCE_DEPS+=("$h")
done

LATEST_SOURCE_DEP_UPDATE=0
for h in "${SOURCE_DEPS[@]}"; do
    if [ -f "$h" ]; then
        h_update=$(stat -c %Y "$h")
        if [ "$h_update" -gt "$LATEST_SOURCE_DEP_UPDATE" ]; then
            LATEST_SOURCE_DEP_UPDATE=$h_update
        fi
    fi
done

BUILD_FILE="./.build/main"
if [ -f "$BUILD_FILE" ]; then
    BUILD_UPDATE=$(stat -c %Y "$BUILD_FILE")
    OLD_CFLAGS=$(readelf --string-dump=.cflags "$BUILD_FILE")
    if [[ "$OLD_CFLAGS" != *"$CFLAGS"* ]]; then
        echo "CFLAGS changed, rebuilding..."
        echo "Old CFLAGS: $OLD_CFLAGS"
        echo "New CFLAGS: $CFLAGS"
        BUILD_UPDATE=0
    elif [ "$FORCE_REBUILD" == "yes" ]; then
        echo "Force rebuild enabled, rebuilding..."
        BUILD_UPDATE=0
    fi
else
    BUILD_UPDATE=0
fi


if [ "$LATEST_SOURCE_DEP_UPDATE" -gt "$BUILD_UPDATE" ]; then
    echo "Changes detected, rebuilding..."

    $CC $CFLAGS -o "$BUILD_FILE" cmd/main.cpp

    objcopy --add-section .cflags=<(echo "$CFLAGS") \
        --set-section-flags .cflags=noload,readonly \
        "$BUILD_FILE"
else
    echo "No changes, skipping build."
fi


echo "Build complete."
if [ "$RUN_CMD" == "yes" ]; then
    ./.build/main "${ARGS[@]}"
fi