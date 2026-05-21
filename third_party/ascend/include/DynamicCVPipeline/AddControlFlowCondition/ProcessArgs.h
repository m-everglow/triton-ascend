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

struct SharedArgInfo {
  int argIndex;           // original iter_arg index
  Value iterArg;          // the original iter_arg value
  int ownerBlockId;       // block_id that "owns" this arg (first in order)
  int newArgIndex;        // new iter_arg index in the new for op
  int nonOwnerBlockId;    // the non-owner block that needs a clone

  SharedArgInfo(int arg, Value val, int owner, int newIdx, int nonOwner)
      : argIndex(arg), iterArg(val), ownerBlockId(owner),
        newArgIndex(newIdx), nonOwnerBlockId(nonOwner) {}
};

class ProcessArgsPass : public PassWrapper<ProcessArgsPass, OperationPass<ModuleOp>> {
public:
  ProcessArgsPass() = default;

  void runOnOperation() override;

  LogicalResult processSharedIterArgs(ModuleOp module);

  llvm::StringRef getArgument() const override
  {
    return "process-args";
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createProcessArgsPass();

} // namespace triton
} // namespace mlir
#endif // TRITON_ASCEND_SSBUF_PROCESS_ARGS_FOR_CONTROL_FLOW_H
