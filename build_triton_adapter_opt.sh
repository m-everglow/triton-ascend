#!/bin/bash
# ============================================================
# triton-adapter-opt 编译脚本
# 用于重新编译 triton-adapter-opt（含所有 SSBUF Pass）
#
# 使用方法:
#   cd /home/zdl/triton-ascend/python/build/cmake.linux-aarch64-cpython-3.11
#   bash /home/zdl/triton-ascend/build_triton_adapter_opt.sh
#
# 前置条件:
#   - cmake 3.29+ (位于 third_party/ascend/AscendNPU-IR/cmake-3.29.3-linux-aarch64/bin/cmake)
#   - 已经完成一次完整 cmake 配置
# ============================================================

set -e

BUILD_DIR="/home/zdl/triton-ascend/python/build/cmake.linux-aarch64-cpython-3.11"
CMAKE_BIN="/home/zdl/triton-ascend/third_party/ascend/AscendNPU-IR/cmake-3.29.3-linux-aarch64/bin/cmake"

echo "[1/4] Reconfigure cmake with TRITON_BUILD_TRITON_ADAPTER_OPT=ON ..."
${CMAKE_BIN} -DTRITON_BUILD_TRITON_ADAPTER_OPT=ON -S /home/zdl/triton-ascend -B ${BUILD_DIR}

echo "[2/4] Build InnerMultibuffer.cpp (incremental) ..."
ninja -C ${BUILD_DIR} third_party/ascend/lib/TritonAffinityOpt/CMakeFiles/TritonAffinityOpt.dir/InnerMultibuffer.cpp.o

echo "[3/4] Link triton-adapter-opt ..."
ninja -C ${BUILD_DIR} third_party/ascend/tools/triton-adapter-opt/triton-adapter-opt

echo "[4/4] Verify --inner-multibuffer pass is available ..."
${BUILD_DIR}/third_party/ascend/tools/triton-adapter-opt/triton-adapter-opt \
    --help 2>&1 | grep -q "inner-multibuffer" \
    && echo "PASS: --inner-multibuffer is registered" \
    || echo "FAIL: --inner-multibuffer not found"

echo ""
echo "Done! Binary: ${BUILD_DIR}/third_party/ascend/tools/triton-adapter-opt/triton-adapter-opt"

# ============================================================
# 快速验证命令:
#   ./third_party/ascend/tools/triton-adapter-opt/triton-adapter-opt \
#       /home/zdl/triton-ascend/test/ssbuf/dag_inner_multibuffer/inner_multibuffer_synthetic.mlir \
#       --inner-multibuffer
# ============================================================
