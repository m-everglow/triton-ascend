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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/HoistOldValueUses.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "HoistOldValueUses";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
    llvm::dbgs() << "\n";                                                      \
  })

using namespace mlir;
using namespace triton;
using namespace CVPipeline;

// A tensor yielded by one branch and an external tensor forwarded by the
// other branch represent the updated and old values of the same if result.
// This is a structural pairing, not a general memory-alias analysis.
static void hoistUses(Value newValue, Value oldValue, Block &block) {
  Operation *definition = newValue.getDefiningOp();
  if (!isa<TensorType>(newValue.getType()) || !definition ||
      definition->getBlock() != &block || newValue == oldValue) {
    return;
  }

  // Snapshot in block order: moving operations must not change traversal order.
  SmallVector<Operation *> users;
  for (Operation *op = definition->getNextNode(); op; op = op->getNextNode()) {
    if (llvm::is_contained(op->getOperands(), oldValue)) {
      users.push_back(op);
    }
  }

  Operation *previousUser = definition;
  for (Operation *user : users) {
    // Terminators must remain at the end of their block.
    if (user->hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }

    llvm::SetVector<Value> inputs;
    inputs.insert(user->operand_begin(), user->operand_end());
    getUsedValuesDefinedAbove(user->getRegions(), inputs);

    Operation *anchor = previousUser;
    for (Value operand : inputs) {
      Operation *producer = operand.getDefiningOp();
      if (producer && producer->getBlock() == &block &&
          anchor->isBeforeInBlock(producer)) {
        anchor = producer;
      }
    }

    if (anchor->getNextNode() != user) {
      LDBG("move " << *user << " after " << *anchor);
      user->moveAfter(anchor);
    }
    previousUser = user;
  }
}

static void hoistOldValueUses(ModuleOp module) {
  module.walk([&](scf::IfOp ifOp) {
    if (!ifOp->hasAttr(CVPipeline::kIf) || ifOp.getElseRegion().empty()) {
      return;
    }
    auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    for (auto [newValue, oldValue] :
         llvm::zip(thenYield.getOperands(), elseYield.getOperands())) {
      if (oldValue.getParentRegion() != &ifOp.getElseRegion()) {
        hoistUses(newValue, oldValue, *ifOp.thenBlock());
      }
      if (newValue.getParentRegion() != &ifOp.getThenRegion()) {
        hoistUses(oldValue, newValue, *ifOp.elseBlock());
      }
    }
  });
}

void HoistOldValueUsesPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("before hoistOldValueUses:\n" << module);
  hoistOldValueUses(module);

  LDBG("after hoistOldValueUses:\n" << module);
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createHoistOldValueUsesPass() {
  return std::make_unique<HoistOldValueUsesPass>();
}

} // namespace triton
} // namespace mlir
