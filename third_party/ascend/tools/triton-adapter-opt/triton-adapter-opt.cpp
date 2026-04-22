/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonAffinityOpt/Passes.h"
#include "TritonToLLVM/Passes.h"
#include "bishengir/InitAllDialects.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

int main(int argc, char **argv) {
  // Register all dialects.
  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
                  mlir::math::MathDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::linalg::LinalgDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::tensor::TensorDialect, mlir::memref::MemRefDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::gpu::GPUDialect>();
  bishengir::registerAllDialects(registry);

  // Register all passes.
  mlir::triton::registerDAGSSBufferPass();
  mlir::triton::registerDAGScopePass();
  mlir::triton::registerDAGSyncPass();
  mlir::triton::registerAddIfControlsPass();
  mlir::triton::registerLifecycleAnalysisPass();
  mlir::triton::registerMultiBufferPass();
  mlir::triton::registerOuterMultiBufferPass();
  mlir::triton::registerTritonToLLVMPass();

  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Triton-Adapter test driver\n", registry));
}