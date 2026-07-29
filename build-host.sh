#!/usr/bin/env bash
# Reproducible host (x86-64 Linux) build of libmoshi + the benchmark tools.
#
# Same disease as the Android tree had: build/CMakeCache.txt recorded
# SentencePiece_INCLUDE_DIR, SentencePiece_LIBRARY and the include flags pointing into a
# session scratchpad under /tmp, which the OS reaped. So the host tree was unbuildable too,
# and the host is where quantization variants get produced and quality-scored.
#
# sentencepiece lives in third_party/ (shared with build-android.sh, same pinned ref) and
# gets built for the host here.
#
# Usage:
#   ./build-host.sh              # configure (if needed) + build
#   ./build-host.sh --clean      # nuke build/ and reconfigure
#   ./build-host.sh --clean-spm  # also rebuild the host sentencepiece
#
# ggml comes from ../voxtral.cpp/build-host — built here if absent.
set -euo pipefail

cd "$(dirname "$0")"
HERE="$PWD"

BUILD_DIR="$HERE/build"
SPM_DIR="$HERE/third_party/sentencepiece"
SPM_BUILD="$SPM_DIR/build-host"
GGML_SRC="$HERE/../voxtral.cpp/ggml"
# Configured with -S on the ggml dir itself, so the libs land in src/, not ggml/src/.
GGML_BUILD="$HERE/../voxtral.cpp/build-host/src"
SPM_REF="${SPM_REF:-v0.2.1}"

CLEAN=0 CLEAN_SPM=0
for arg in "$@"; do
  case "$arg" in
    --clean)     CLEAN=1 ;;
    --clean-spm) CLEAN=1; CLEAN_SPM=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# ---- ggml for the host ------------------------------------------------------------------
if [ ! -f "$GGML_BUILD/libggml.so" ]; then
  echo "== building host ggml"
  cmake -B "$HERE/../voxtral.cpp/build-host" -S "$GGML_SRC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_OPENMP=OFF
  cmake --build "$HERE/../voxtral.cpp/build-host" -j"$(nproc)"
fi

# ---- sentencepiece for the host ---------------------------------------------------------
[ "$CLEAN_SPM" = 1 ] && rm -rf "$SPM_BUILD"

if [ ! -d "$SPM_DIR/.git" ]; then
  echo "== cloning sentencepiece $SPM_REF"
  rm -rf "$SPM_DIR"
  git clone --depth 1 --branch "$SPM_REF" https://github.com/google/sentencepiece "$SPM_DIR"
fi

if [ ! -f "$SPM_BUILD/src/libsentencepiece.a" ]; then
  echo "== building host sentencepiece"
  cmake -B "$SPM_BUILD" -S "$SPM_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DSPM_ENABLE_SHARED=OFF \
    -DSPM_BUILD_TEST=OFF \
    -DSPM_ENABLE_TCMALLOC=OFF
  cmake --build "$SPM_BUILD" -j"$(nproc)" --target sentencepiece-static
fi

[ -f "$SPM_BUILD/src/libsentencepiece.a" ] || {
  echo "sentencepiece static lib not produced; look in $SPM_BUILD/src" >&2; exit 1; }

# v0.2.1 vendors abseil into libsentencepiece.a; older layouts ship it as separate archives.
mapfile -t ABSL < <(find "$SPM_BUILD/third_party/abseil-cpp" -name 'libabsl_*.a' 2>/dev/null | sort)
if [ "${#ABSL[@]}" -eq 0 ]; then
  LINK_FLAGS=""
else
  LINK_FLAGS="-Wl,--start-group ${ABSL[*]} -Wl,--end-group"
fi

SPM_INCLUDES="-I$SPM_DIR -I$SPM_DIR/third_party/abseil-cpp -I$SPM_DIR/third_party/protobuf-lite"

# ---- libmoshi + tools ------------------------------------------------------------------
[ "$CLEAN" = 1 ] && rm -rf "$BUILD_DIR"

if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  cmake -B "$BUILD_DIR" -S "$HERE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMOSHI_BUILD_TOOLS=ON \
    -DMOSHI_TOOLS_MEDIA=OFF \
    -DCMAKE_CXX_FLAGS="$SPM_INCLUDES" \
    -DCMAKE_SHARED_LINKER_FLAGS="$LINK_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="$LINK_FLAGS" \
    -DGGML_INCLUDE_DIR="$GGML_SRC/include" \
    -DGGML_LIBRARY_DIR="$GGML_BUILD" \
    -DGGML_LIBRARY="$GGML_BUILD/libggml.so" \
    -DGGML_BASE_LIBRARY="$GGML_BUILD/libggml-base.so" \
    -DGGML_CPU_LIBRARY="$GGML_BUILD/libggml-cpu.so" \
    -DSentencePiece_INCLUDE_DIR="$SPM_DIR/src" \
    -DSentencePiece_LIBRARY="$SPM_BUILD/src/libsentencepiece.a" \
    -DSentencePiece_LIBRARY_DIR="$SPM_BUILD/src"
fi

cmake --build "$BUILD_DIR" -j"$(nproc)"

echo
echo "built:"
ls -la "$BUILD_DIR"/bin/libmoshi.so "$BUILD_DIR"/bin/requantize_gguf "$BUILD_DIR"/bin/stt_bench
