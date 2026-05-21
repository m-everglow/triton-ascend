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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/ProcessArgs.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"

static constexpr const char *DEBUG_TYPE = "ProcessArgs";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::dbgs() << __VA_ARGS__; \
  llvm::dbgs() << "\n"; \
})

using namespace llvm;
using namespace mlir;
using namespace triton;

// Find the mapping between block_ids and iter_args
static LogicalResult collectArgIndexToBlockIds(scf::ForOp forOp,
    llvm::DenseMap<int, llvm::DenseSet<int>> &argIndexToBlockIds)
{
  Block *body = forOp.getBody();
  for (Operation &op : body->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    if (!blockIdAttr)
      continue;
    int blockId = blockIdAttr.getInt();

    for (OpOperand &operand : op.getOpOperands()) {
      Value v = operand.get();
      for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
        if (v == forOp.getRegionIterArgs()[i]) {
          argIndexToBlockIds[i].insert(blockId);
        }
      }
    }
  }
  return success();
}

// Find shared args and create SharedArgInfo for each non-owner block
static LogicalResult collectSharedArgsInfo(scf::ForOp forOp,
    const llvm::DenseMap<int, llvm::DenseSet<int>> &argIndexToBlockIds,
    SmallVector<SharedArgInfo> &sharedArgsInfo)
{
  SmallVector<int> idsInOrder = getBlockIdsInOrder(forOp);
  int extraArgCount = 0;

  for (auto &p : argIndexToBlockIds) {
    int argIndex = p.first;
    const llvm::DenseSet<int> &blockIds = p.second;

    if (blockIds.size() <= 1)
      continue;

    // Find owner block (first in order)
    int ownerBlockId = -1;
    for (int id : idsInOrder) {
      if (blockIds.contains(id)) {
        ownerBlockId = id;
        break;
      }
    }
    if (ownerBlockId == -1)
      continue;

    int baseIndex = forOp.getNumRegionIterArgs() + extraArgCount;

    // Create SharedArgInfo for each non-owner block
    for (int bid : blockIds) {
      if (bid == ownerBlockId)
        continue;
      sharedArgsInfo.emplace_back(argIndex, forOp.getRegionIterArgs()[argIndex],
                                ownerBlockId, baseIndex++, bid);
      extraArgCount++;
    }
  }
  return success();
}

// Find the computation op in owner block whose result goes to yield
static LogicalResult findCompOpInOwnerBlock(Block *body, int ownerBlockId,
    Value iterArg, Operation *&compOp)
{
  compOp = nullptr;
  for (Operation &op : body->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    if (!blockIdAttr || blockIdAttr.getInt() != ownerBlockId)
      continue;

    bool usesIterArg = llvm::any_of(op.getOperands(),
                                     [&](Value v) { return v == iterArg; });
    if (!usesIterArg)
      continue;

    for (Value result : op.getResults()) {
      for (OpOperand &use : result.getUses()) {
        if (isa<scf::YieldOp>(use.getOwner())) {
          compOp = &op;
          return success();
        }
      }
    }
  }
  LDBG("[ERROR]: Could not find comp op for blockId: " << ownerBlockId);
  return failure();
}

// Collect computation chain via backward traversal
static LogicalResult collectCompChain(Operation *compOp, scf::ForOp forOp,
    llvm::DenseSet<Operation *> &chainOps)
{
  chainOps.clear();
  if (!compOp)
    return success();

  SmallVector<Operation *> worklist{compOp};
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (chainOps.contains(op))
      continue;
    chainOps.insert(op);

    for (Value operand : op->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        if (defOp->getParentOp() == forOp && !chainOps.contains(defOp)) {
          worklist.push_back(defOp);
        }
      }
    }
  }
  return success();
}

// Find computation chains for each shared arg
static LogicalResult findComputationChains(scf::ForOp forOp,
    const SmallVector<SharedArgInfo> &sharedArgsInfo,
    llvm::DenseMap<int, Operation *> &sharedArgToCompOp,
    llvm::DenseMap<int, llvm::DenseSet<Operation *>> &sharedArgToChainOps)
{
  Block *body = forOp.getBody();
  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    if (sharedArgToCompOp.contains(argIndex))
      continue;

    Value iterArg = forOp.getRegionIterArgs()[argIndex];
    Operation *compOp = nullptr;
    if (failed(findCompOpInOwnerBlock(body, info.ownerBlockId, iterArg, compOp))) {
      return failure();
    }

    if (!compOp) {
      LDBG("[ERROR]: Could not find comp op for arg index: " << argIndex);
      return failure();
    }

    sharedArgToCompOp[argIndex] = compOp;
    llvm::DenseSet<Operation *> chainOps;
    if (failed(collectCompChain(compOp, forOp, chainOps))) {
      return failure();
    }
    sharedArgToChainOps[argIndex] = chainOps;
  }
  return success();
}

// Get last operation in block belonging to the specified block_id
static Operation *findLastOpInBlock(Block *newBlock, int blockId)
{
  Operation *lastOp = nullptr;
  for (Operation &op : newBlock->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    if (blockIdAttr && blockIdAttr.getInt() == blockId) {
      lastOp = &op;
    }
  }
  return lastOp;
}

// Clone the computation chain for a non-owner block
static LogicalResult cloneChainForBlock(const SharedArgInfo &info,
    Block *newBlock, Operation *compOp,
    const llvm::DenseSet<Operation *> &chainOps,
    Value originalBlockArg, Value newBlockArg,
    Value &clonedResult)
{
  clonedResult = nullptr;
  if (!compOp || chainOps.empty())
    return success();

  SmallVector<Operation *> sortedChain(chainOps.begin(), chainOps.end());
  if (failed(topologicalSort(sortedChain)))
    return failure();

  // Find insertion point after last op of this block
  Operation *lastOpInBlock = findLastOpInBlock(newBlock, info.nonOwnerBlockId);
  OpBuilder cloneBuilder(newBlock, newBlock->end());
  if (lastOpInBlock) {
    cloneBuilder.setInsertionPointAfter(lastOpInBlock);
  }

  // Map the original value to cloned value
  IRMapping argMapper;
  argMapper.map(originalBlockArg, newBlockArg);

  IRMapping resultMapper;
  for (Operation *op : sortedChain) {
    IRMapping opMapper;
    for (OpOperand &operand : op->getOpOperands()) {
      Value oldVal = operand.get();
      Value newVal;
      if (resultMapper.contains(oldVal)) {
        newVal = resultMapper.lookup(oldVal);
      } else if (argMapper.contains(oldVal)) {
        newVal = argMapper.lookup(oldVal);
      } else {
        newVal = oldVal;
      }
      opMapper.map(oldVal, newVal);
    }

    if (resultMapper.contains(op->getResult(0)))
      continue;

    Operation *cloned = cloneBuilder.clone(*op, opMapper);
    cloned->setAttr("ssbuffer.block_id", cloneBuilder.getI32IntegerAttr(info.nonOwnerBlockId));
    cloned->setAttr("ssbuffer.arg", cloneBuilder.getI32IntegerAttr(info.argIndex));

    resultMapper.map(op->getResult(0), cloned->getResult(0));
    cloneBuilder.setInsertionPointAfter(cloned);
  }

  clonedResult = resultMapper.lookupOrDefault(compOp->getResult(0));
  return success();
}

// Migrate body, redirect old block args and move ops
static LogicalResult migrateBody(scf::ForOp forOp, scf::ForOp newForOp,
    SmallVector<Value> &oldBlockArgs)
{
  Block *oldBlock = forOp.getBody();
  Block *newBlock = newForOp.getBody();

  for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
    oldBlockArgs.push_back(oldBlock->getArgument(i));
  }

  for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
    oldBlock->getArgument(i).replaceAllUsesWith(newBlock->getArgument(i));
  }

  for (Operation &op : llvm::make_early_inc_range(oldBlock->without_terminator())) {
    op.moveBefore(newBlock, newBlock->end());
  }
  return success();
}

// Build new yield with original operands + cloned results
static LogicalResult buildNewYield(scf::ForOp newForOp, Block *newBlock,
    Block *oldBlock, const SmallVector<Value> &clonedResults)
{
  auto oldYield = cast<scf::YieldOp>(oldBlock->getTerminator());
  SmallVector<Value> yieldOperands;

  for (unsigned i = 0; i < oldYield.getNumOperands(); ++i) {
    yieldOperands.push_back(oldYield.getOperand(i));
  }
  for (auto &result : clonedResults) {
    yieldOperands.push_back(result);
  }

  OpBuilder builder(newForOp);
  builder.setInsertionPointToEnd(newBlock);
  builder.create<scf::YieldOp>(newForOp.getLoc(), yieldOperands);
  oldYield.erase();
  return success();
}

// Create new for op with extra iter_args
static LogicalResult createNewForOp(scf::ForOp forOp,
    const SmallVector<SharedArgInfo> &sharedArgsInfo,
    scf::ForOp &newForOp)
{
  OpBuilder builder(forOp);
  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                  forOp.getInitArgs().end());

  for (auto &info : sharedArgsInfo) {
    newInitArgs.push_back(forOp.getInitArgs()[info.argIndex]);
  }

  newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);

  for (auto &attr : forOp->getAttrs()) {
    newForOp->setAttr(attr.getName(), attr.getValue());
  }
  return success();
}

// Replace uses of original iter_arg with new iter_arg in non-owner block
static LogicalResult replaceIterArgUses(Block *newBlock, OpBuilder &builder,
    int nonOwnerBlockId, int argIndex,
    Value originalArg, Value newArg)
{
  for (Operation &op : newBlock->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    if (!blockIdAttr || blockIdAttr.getInt() != nonOwnerBlockId)
      continue;

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (op.getOperand(i) == originalArg) {
        op.setOperand(i, newArg);
        op.setAttr("ssbuffer.arg", builder.getI32IntegerAttr(argIndex));
      }
    }
  }
  return success();
}

// Clone chains for all non-owner blocks
static LogicalResult cloneAllChains(scf::ForOp forOp,
    const SmallVector<SharedArgInfo> &sharedArgsInfo,
    const llvm::DenseMap<int, Operation *> &sharedArgToCompOp,
    const llvm::DenseMap<int, llvm::DenseSet<Operation *>> &sharedArgToChainOps,
    SmallVector<Value> &clonedResults)
{
  clonedResults.clear();
  Block *newBlock = forOp.getBody();
  OpBuilder builder(forOp);

  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    Operation *compOp = sharedArgToCompOp.lookup(argIndex);
    if (!compOp)
      continue;

    const llvm::DenseSet<Operation *> &chainOps = sharedArgToChainOps.lookup(argIndex);
    if (chainOps.empty())
      continue;

    Value originalBlockArg = newBlock->getArgument(info.argIndex + 1);
    Value newBlockArg = newBlock->getArgument(info.newArgIndex + 1);

    Value clonedResult;
    if (failed(cloneChainForBlock(info, newBlock, compOp, chainOps,
                                  originalBlockArg, newBlockArg, clonedResult))) {
      return failure();
    }
    if (clonedResult)
      clonedResults.push_back(clonedResult);

    if (failed(replaceIterArgUses(newBlock, builder, info.nonOwnerBlockId, info.argIndex,
                                  originalBlockArg, newBlockArg))) {
      return failure();
    }
  }
  return success();
}

// Main function to process shared iter_args in a forOp
static LogicalResult processSharedIterArgsInForOp(scf::ForOp forOp)
{
  Block *body = forOp.getBody();
  if (!body || !body->mightHaveTerminator()) {
    LDBG("[Error]: forOp body is invalid or has no terminator\n");
    return failure();
  }

  // Step 1: Analyze the mapping relationship between iter_args and block_ids
  llvm::DenseMap<int, llvm::DenseSet<int>> argIndexToBlockIds;
  if (failed(collectArgIndexToBlockIds(forOp, argIndexToBlockIds))) {
    return failure();
  }

  SmallVector<SharedArgInfo> sharedArgsInfo;
  if (failed(collectSharedArgsInfo(forOp, argIndexToBlockIds, sharedArgsInfo))) {
    return failure();
  }

  if (sharedArgsInfo.empty())
    return success();

  LDBG("[INFO]: Found " << sharedArgsInfo.size() << " shared iter_args to process\n");

  // Step 2: Find computation chains
  llvm::DenseMap<int, Operation *> sharedArgToCompOp;
  llvm::DenseMap<int, llvm::DenseSet<Operation *>> sharedArgToChainOps;
  if (failed(findComputationChains(forOp, sharedArgsInfo,
                                   sharedArgToCompOp, sharedArgToChainOps))) {
    return failure();
  }

  // Step 3: Create new for op
  scf::ForOp newForOp;
  if (failed(createNewForOp(forOp, sharedArgsInfo, newForOp))) {
    return failure();
  }

  // Step 4: Migrate body
  SmallVector<Value> oldBlockArgs;
  if (failed(migrateBody(forOp, newForOp, oldBlockArgs))) {
    return failure();
  }

  // Step 5: Clone chains
  SmallVector<Value> clonedResults;
  if (failed(cloneAllChains(newForOp, sharedArgsInfo,
                            sharedArgToCompOp, sharedArgToChainOps,
                            clonedResults))) {
    return failure();
  }

  // Step 6: Build yield and cleanup
  if (failed(buildNewYield(newForOp, newForOp.getBody(), forOp.getBody(), clonedResults))) {
    return failure();
  }

  if (forOp.getNumResults() > 0) {
    SmallVector<Value> newResults;
    for (unsigned i = 0; i < forOp.getNumResults(); ++i)
      newResults.push_back(newForOp.getResult(i));
    forOp.replaceAllUsesWith(newResults);
  }
  forOp.erase();

  return success();
}

LogicalResult ProcessArgsPass::processSharedIterArgs(ModuleOp module)
{
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr("ssbuffer.main_loop")) {
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      return WalkResult::interrupt();
    }

    if (failed(processSharedIterArgsInForOp(forOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }
  return success();
}

void ProcessArgsPass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("before processArgs:\n" << module << "\n");

  if (failed(processSharedIterArgs(module))) {
    signalPassFailure();
    return;
  }

  LDBG("after processArgs:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createProcessArgsPass()
{
  return std::make_unique<ProcessArgsPass>();
}

} // namespace triton
} // namespace mlir
