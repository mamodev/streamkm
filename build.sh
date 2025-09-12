set -e

mkdir -p .build

CC=clang++
CFLAGS="--std=c++20 -I. "
CFLAGS+=" -O3 -ffast-math -march=native -DDISABLE_PASSERT"
# CFLAGS+=" -O0 -g -fsanitize=address -fno-omit-frame-pointer"


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
else
    BUILD_UPDATE=0
fi

if [ "$LATEST_SOURCE_DEP_UPDATE" -gt "$BUILD_UPDATE" ]; then
    echo "Changes detected, rebuilding..."

    $CC $CFLAGS -o "$BUILD_FILE" cmd/main.cpp

    # g++ -J8 -O3 -ffast-math -march=native --std=c++20 -I. -o "$BUILD_FILE" cmd/main.cpp
    # -lopenblas
    # g++ --std=c++20 -I. -o main main.cpp -lopenblas
    # g++ -O0 -g -fsanitize=address --std=c++20 -I. -o main main.cpp
    # g++ -O3 -ffast-math -march=native --std=c++20 -I. -o main main.cpp   

    # g++ -O3 -ffast-math -march=native --std=c++20 -I. -o main main.cpp        -DDISABLE_PASSERT

    # clang++ -O3 -ffast-math -march=native --std=c++20 -I. -o main main.cpp    -DDISABLE_PASSERT -lopenblas

    # clang++ -O2 -g -fno-omit-frame-pointer -ffast-math -march=native --std=c++20 -I. -o main main.cpp -DDISABLE_PASSERT
else
    echo "No changes, skipping build."
fi


echo "Build complete, running..."
./.build/main "$@"
