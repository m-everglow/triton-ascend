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

#ifndef TRITON_ADAPTER_TRITON_AFFINITY_OPTIMIZATION_PASSES_H
#define TRITON_ADAPTER_TRITON_AFFINITY_OPTIMIZATION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
// Forward declarations.
class ModuleOp;

namespace triton {

/// Creates a pass to convert Triton dialect to Annotation dialect.
std::unique_ptr<OperationPass<ModuleOp>> createDAGSSBufferPass();

/// Creates a pass for §6.1: Compute Block Partition (C/V type + block ID + VFFusion).
std::unique_ptr<OperationPass<ModuleOp>> createComputeBlockPartitionPass();

/// Creates a pass for §6.5: AddControlFlow (scf.if per block_id + block counters + flag conditions).
std::unique_ptr<OperationPass<ModuleOp>> createAddControlFlowPass();

std::unique_ptr<OperationPass<ModuleOp>> createDAGSyncPass();

std::unique_ptr<OperationPass<ModuleOp>> createDAGScopePass();

std::unique_ptr<OperationPass<ModuleOp>> createAddIfControlsPass();

std::unique_ptr<OperationPass<ModuleOp>> createLifecycleAnalysisPass();

std::unique_ptr<OperationPass<ModuleOp>> createMultiBufferPass();

std::unique_ptr<OperationPass<ModuleOp>> createOuterMultiBufferPass();

#define GEN_PASS_REGISTRATION
#include "ascend/include/TritonAffinityOpt/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TRITON_AFFINITY_OPTIMIZATION_PASSES_H
