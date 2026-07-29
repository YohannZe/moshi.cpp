#!/usr/bin/env bash
# Reproducible Android (arm64-v8a) build of libmoshi.so — the kyutai stt-1b engine.
#
# Exists because the previous build-android/ was hand-configured with
# SentencePiece_LIBRARY and a ~90-archive abseil --start-group link line pointing into a
# session scratchpad under /tmp. The OS reaped it, and the tree became unbuildable — after
# having silently produced a stale .so that got measured as if it were current.
#
# So sentencepiece lives in third_party/ here, is built by this script, and the abseil
# archives are discovered by glob instead of being pasted in by hand.
#
# Usage:
#   ./build-android.sh              # configure (if needed) + build
#   ./build-android.sh --clean      # nuke build-android/ and reconfigure
#   ./build-android.sh --clean-spm  # also rebuild sentencepiece
#
# ggml comes from ../voxtral.cpp/build-android — build that first (../build.sh does).
set -euo pipefail

cd "$(dirname "$0")"
HERE="$PWD"

ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
NDK_VERSION="${NDK_VERSION:-27.0.12077973}"
NDK="$ANDROID_HOME/ndk/$NDK_VERSION"
TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"

BUILD_DIR="$HERE/build-android"
SPM_DIR="$HERE/third_party/sentencepiece"
SPM_BUILD="$SPM_DIR/build-android"
GGML_SRC="$HERE/../voxtral.cpp/ggml"
GGML_BUILD="$HERE/../voxtral.cpp/build-android/ggml/src"

# Pinned: the abseil-bundling layout this script globs for arrived in 0.2.x, and an
# unpinned master has already broken this build once.
SPM_REF="${SPM_REF:-v0.2.1}"

CLEAN=0 CLEAN_SPM=0
for arg in "$@"; do
  case "$arg" in
    --clean)     CLEAN=1 ;;
    --clean-spm) CLEAN=1; CLEAN_SPM=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

[ -f "$TOOLCHAIN" ] || { echo "NDK toolchain not found: $TOOLCHAIN" >&2; exit 1; }
[ -d "$GGML_BUILD" ] || { echo "ggml not built: $GGML_BUILD" >&2
                          echo "  run ../voxtral.cpp/build-android.sh first (or ../build.sh)" >&2
                          exit 1; }

# ---- sentencepiece (static, arm64) -----------------------------------------------------
[ "$CLEAN_SPM" = 1 ] && rm -rf "$SPM_BUILD"

if [ ! -d "$SPM_DIR/.git" ]; then
  echo "== cloning sentencepiece $SPM_REF"
  rm -rf "$SPM_DIR"
  git clone --depth 1 --branch "$SPM_REF" https://github.com/google/sentencepiece "$SPM_DIR"
fi

if [ ! -f "$SPM_BUILD/src/libsentencepiece.a" ]; then
  echo "== building sentencepiece for arm64-v8a"
  cmake -B "$SPM_BUILD" -S "$SPM_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPM_ENABLE_SHARED=OFF \
    -DSPM_BUILD_TEST=OFF \
    -DSPM_ENABLE_TCMALLOC=OFF
  cmake --build "$SPM_BUILD" -j"$(nproc)" --target sentencepiece-static
  # The static target may be named either way depending on SPM_ENABLE_SHARED handling.
  [ -f "$SPM_BUILD/src/libsentencepiece.a" ] || \
    cmake --build "$SPM_BUILD" -j"$(nproc)"
fi

[ -f "$SPM_BUILD/src/libsentencepiece.a" ] || {
  echo "sentencepiece static lib not produced; look in $SPM_BUILD/src" >&2; exit 1; }

# ---- the abseil archives sentencepiece needs, by glob rather than by hand --------------
# Order matters for a static link, so wrap the lot in --start-group/--end-group and let the
# linker iterate instead of trying to topologically sort ~90 archives.
mapfile -t ABSL < <(find "$SPM_BUILD/third_party/abseil-cpp" -name 'libabsl_*.a' 2>/dev/null | sort)
if [ "${#ABSL[@]}" -eq 0 ]; then
  # Some sentencepiece builds fold abseil straight into libsentencepiece.a.
  echo "== no separate abseil archives; assuming they are inside libsentencepiece.a"
  LINK_FLAGS="-llog"
else
  echo "== ${#ABSL[@]} abseil archives"
  LINK_FLAGS="-llog -Wl,--start-group ${ABSL[*]} -Wl,--end-group"
fi

SPM_INCLUDES="-I$SPM_DIR -I$SPM_DIR/third_party/abseil-cpp -I$SPM_DIR/third_party/protobuf-lite"

# ---- libmoshi.so ----------------------------------------------------------------------
[ "$CLEAN" = 1 ] && rm -rf "$BUILD_DIR"

if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  cmake -B "$BUILD_DIR" -S "$HERE" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release \
    -DMOSHI_BUILD_TOOLS=ON \
    -DMOSHI_TOOLS_MEDIA=OFF \
    -DCMAKE_CXX_FLAGS="$SPM_INCLUDES" \
    -DCMAKE_SHARED_LINKER_FLAGS="$LINK_FLAGS" \
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
ls -la "$BUILD_DIR/bin/libmoshi.so"
