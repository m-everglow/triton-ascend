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

#ifndef TRITON_ASCEND_SSBUF_PROCESS_ARGS_FOR_CONTROL_FLOW_H
#define TRITON_ASCEND_SSBUF_PROCESS_ARGS_FOR_CONTROL_FLOW_H
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

struct ControlFlowConditionInfo;

// For each shared iter_arg, we need to track:
// - Which block_ids use it
// - Who is the owner (first block_id in order)
// - For each non-owner block, what new iter_arg index to use
struct SharedArgInfo {
  int argIndex;
  Value iterArg;
  int ownerBlockId;
  int newArgIndex;
  int nonOwnerBlockId;

  SharedArgInfo(int arg, int owner, int newIdx, int nonOwner)
      : argIndex(arg), iterArg(Value()), ownerBlockId(owner),
        newArgIndex(newIdx), nonOwnerBlockId(nonOwner) {}
};

class ProcessArgsPass : public PassWrapper<ProcessArgsPass, OperationPass<ModuleOp>> {
 public:
  ProcessArgsPass() = default;

  void runOnOperation() override;

  LogicalResult processSharedIterArgs(ModuleOp module);

  // Records the original iter_args of every scf.while op with main_loop attr
  // before any other processing happens. Used later to identify which
  // iter_args are referenced by scf.condition and need per-block update
  // chains.
  void recordOriginalWhileIterArgs(ModuleOp module);

  // For every scf.while op with main_loop attr, clones the update chain of
  // each iter_arg used in scf.condition into the end of every block (group of
  // consecutive ops sharing the same ssbuffer.block_id) in the do region.
  // Each clone is annotated with `ssbuffer.while_arg = <original index>` and
  // `ssbuffer.block_id = <target block id>`. A fresh scf.while op is built
  // with one additional iter_arg per (block_id, original-iter_arg) pair, and
  // a mapping from while op -> block_id -> (new_arg_idx, old_arg_idx) is
  // recorded in the ControlFlowConditionInfo for downstream passes.
  LogicalResult processWhileIterArgs(ModuleOp module);

  // Per-whileOp driver for processWhileIterArgs; needs access to
  // originalWhileIterArgIndices (recorded by recordOriginalWhileIterArgs).
  LogicalResult processWhileIterArgsInWhileOp(scf::WhileOp whileOp,
                                              ControlFlowConditionInfo *info);

  // Per-op driver for the existing shared-iter_args processing; member
  // function so it can update originalWhileIterArgIndices when an
  // scf.while op is replaced.
  LogicalResult processSharedIterArgsInLoop(Operation *op,
                                            ControlFlowConditionInfo *info);

  void setConditionInfo(ControlFlowConditionInfo *info_) { info = info_; }

  llvm::StringRef getArgument() const override { return "process-args"; }

  ControlFlowConditionInfo *info = nullptr;

  // Original iter_args (indices into the original scf.while's iter_arg list)
  // for each scf.while op with main_loop attr, captured at the start of
  // ProcessArgs. Used to identify iter_args referenced by scf.condition.
  llvm::DenseMap<scf::WhileOp, SmallVector<unsigned>> originalWhileIterArgIndices;

  // Local copy of whileBlockArgMap maintained by processWhileIterArgs. This
  // is filled in addition to info->whileBlockArgMap so the mapping is
  // observable when the pass is run standalone (--process-args) without
  // going through AddControlFlowConditionPass, where `info` may be null.
  llvm::DenseMap<scf::WhileOp, llvm::DenseMap<int, llvm::DenseMap<int, int>>>
      localWhileBlockArgMap;
};

std::unique_ptr<OperationPass<ModuleOp>> createProcessArgsPass();

} // namespace triton
} // namespace mlir
#endif // TRITON_ASCEND_SSBUF_PROCESS_ARGS_FOR_CONTROL_FLOW_H
