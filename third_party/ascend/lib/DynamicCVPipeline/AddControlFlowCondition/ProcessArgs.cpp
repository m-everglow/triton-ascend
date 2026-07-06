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
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"

static constexpr const char *DEBUG_TYPE = "ProcessArgs";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::dbgs() << __VA_ARGS__; \
  llvm::dbgs() << "\n"; \
})

using namespace mlir;
using namespace triton;

// Collects mapping from iter_arg index to block_ids that use it.
// For each iter_arg, tracks which block_ids reference it in their operations.
// `body` is the loop body (scf.for body or scf.while after-region body).
// `ivOffset` is 1 for scf.for (the induction variable sits at block arg 0;
// iter_args start at 1) and 0 for scf.while (no IV; iter_args start at 0).
// We index by the iter_arg's position in the iter_args list — i.e.
// argNumber - ivOffset — so callers can address entries by iter_arg index
// rather than absolute block-arg position. This matters for scf.for, where
// the iter_arg index is what yieldOp.getOperand(i) expects. We compare
// against body->getArguments() directly (rather than loopOp.getRegionIterArgs())
// because for scf.while those are different blocks' args (before vs after) and
// SSA equality is block-local.
static LogicalResult collectArgIndexToBlockIds(
    Block *body,
    unsigned ivOffset,
    llvm::DenseMap<int, llvm::DenseSet<int>> &argIndexToBlockIds)
{
  if (!body || !body->mightHaveTerminator()) {
    LDBG("[Error]: loop body is invalid or has no terminator\n");
    return failure();
  }

  for (Operation &op : body->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
    if (!blockIdAttr) continue;
    int blockId = blockIdAttr.getInt();

    for (OpOperand &operand : op.getOpOperands()) {
      Value v = operand.get();
      for (BlockArgument iterArg : body->getArguments()) {
        int argIdx = iterArg.getArgNumber();
        if (argIdx < (int)ivOffset) {
          // scf.for's IV at block arg 0 — never an iter_arg.
          continue;
        }
        // Skip tensor-type iter_args, only process scalar and index types
        if (mlir::isa<TensorType>(iterArg.getType())) {
          continue;
        }
        if (v == iterArg) {
          argIndexToBlockIds[argIdx - (int)ivOffset].insert(blockId);
        }
      }
    }
  }
  return success();
}

// Finds iter_args used by multiple block_ids (shared args).
// Determines owner block (first in order) and creates SharedArgInfo for each non-owner.
// Each non-owner block gets its own extra iter_arg.
static LogicalResult findSharedArgs(
    const llvm::DenseMap<int, llvm::DenseSet<int>> &argIndexToBlockIds,
    const SmallVector<int> &idsInOrder,
    SmallVector<SharedArgInfo> &sharedArgsInfo)
{
  int extraArgCount = 0;
  for (auto &p : argIndexToBlockIds) {
    int argIndex = p.first;
    const llvm::DenseSet<int> &blockIds = p.second;

    if (blockIds.size() <= 1) continue;

    int ownerBlockId = -1;
    for (int id : idsInOrder) {
      if (blockIds.contains(id)) {
        ownerBlockId = id;
        break;
      }
    }
    if (ownerBlockId == -1) continue;

    // Each non-owner block for this argIndex gets its own extra iter_arg
    for (int bid : blockIds) {
      if (bid != ownerBlockId) {
        sharedArgsInfo.push_back(
            SharedArgInfo(argIndex, ownerBlockId, extraArgCount, bid));
        extraArgCount++;
      }
    }
  }
  return success();
}

// Finds the computation operation in owner block that produces the iter_arg value.
// compOp is the defining op of the iter_arg in the scf.yield operand list. The
// caller-provided body is the loop body (forOp body or whileOp after-body);
// the iter_arg's position in the yield matches the iter_arg's position in the
// region argument list, so this is op-agnostic.
static LogicalResult findCompOpInOwnerBlock(
    Block *body,
    const SharedArgInfo &info,
    Operation *&compOp)
{
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
  Value yieldArg = yieldOp.getOperand(info.argIndex);

  if (auto *defOp = yieldArg.getDefiningOp()) {
    compOp = defOp;
    return success();
  }

  return failure();
}

// Collects all operations in the computation chain by backward traversal from compOp.
// Builds the dependency graph needed to clone the computation for non-owner blocks.
// `loopOp` is the main-loop op (scf.for or scf.while) and scopes the walk to
// ops inside its body.
static void collectChainOps(
    Operation *loopOp,
    Operation *compOp,
    llvm::DenseSet<Operation*> &chainOps)
{
  SmallVector<Operation*> worklist;
  worklist.push_back(compOp);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (chainOps.contains(op)) continue;
    chainOps.insert(op);

    for (Value operand : op->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        if (defOp->getParentOp() == loopOp && !chainOps.contains(defOp)) {
          worklist.push_back(defOp);
        }
      }
    }
  }
}

// Builds computation info (compOp and chainOps) for each shared arg. `loopOp`
// is the main-loop op (scf.for or scf.while) and scopes the chain walk;
// `body` is the loop body (forOp body or whileOp after-body) and is used to
// locate the scf.yield terminator for finding the compOp.
static LogicalResult buildCompInfoForSharedArgs(
    Operation *loopOp,
    Block *body,
    SmallVector<SharedArgInfo> &sharedArgsInfo,
    llvm::DenseMap<int, Operation*> &sharedArgToCompOp,
    llvm::DenseMap<int, llvm::DenseSet<Operation*>> &sharedArgToChainOps)
{
  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    if (sharedArgToCompOp.contains(argIndex)) continue;

    Operation *compOp = nullptr;
    if (failed(findCompOpInOwnerBlock(body, info, compOp))) {
      continue;
    }

    sharedArgToCompOp[argIndex] = compOp;

    llvm::DenseSet<Operation*> chainOps;
    collectChainOps(loopOp, compOp, chainOps);
    sharedArgToChainOps[argIndex] = chainOps;
  }
  return success();
}

// Creates a new scf.for op with extra iter_args for shared arguments.
// Copies attributes from the original for op.
// Each SharedArgInfo entry (non-owner block) gets its own extra iter_arg.
static scf::ForOp createNewForOp(
    scf::ForOp forOp,
    const SmallVector<SharedArgInfo> &sharedArgsInfo)
{
  OpBuilder builder(forOp);
  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(), forOp.getInitArgs().end());

  // Each non-owner block gets its own extra iter_arg
  for (auto &info : sharedArgsInfo) {
    newInitArgs.push_back(forOp.getInitArgs()[info.argIndex]);
  }

  scf::ForOp newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);

  for (auto &attr : forOp->getAttrs()) {
    newForOp->setAttr(attr.getName(), attr.getValue());
  }
  return newForOp;
}

// Migrates operations from old block to new block.
// Redirects block arguments to new block arguments and moves all ops.
static void migrateBody(Block *oldBlock, Block *newBlock)
{
  for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
    oldBlock->getArgument(i).replaceAllUsesWith(newBlock->getArgument(i));
  }

  for (Operation &op : llvm::make_early_inc_range(oldBlock->without_terminator())) {
    op.moveBefore(newBlock, newBlock->end());
  }
}

// Clones the computation chain for a non-owner block.
// Topologically sorts the chain and clones each op with remapped operands.
// argRemapping: maps migrated iter_arg Value -> new extra iter_arg Value.
// resultMapper: maps original op results -> cloned op results.
// clonedArgIdx: unique index for this non-owner block's clone (used as ssbuffer.arg).
static LogicalResult cloneChainForBlock(
    SharedArgInfo &info,
    Operation *compOp,
    const llvm::DenseSet<Operation*> &chainOps,
    Block *newBlock,
    IRMapping &argRemapping,
    OpBuilder &cloneBuilder,
    IRMapping &resultMapper,
    int clonedArgIdx)
{
  if (!compOp || chainOps.empty()) {
    return failure();
  }

  SmallVector<Operation *> sortedChain(chainOps.begin(), chainOps.end());
  if (failed(topologicalSort(sortedChain))) {
    return failure();
  }

  for (Operation *op : sortedChain) {
    IRMapping opMapper;
    for (OpOperand &operand : op->getOpOperands()) {
      Value oldVal = operand.get();
      Value newVal = oldVal;
      if (argRemapping.contains(oldVal)) {
        newVal = argRemapping.lookup(oldVal);
      } else if (resultMapper.contains(oldVal)) {
        // Operand is a result from earlier in the owner chain, use cloned result
        newVal = resultMapper.lookup(oldVal);
      }
      opMapper.map(oldVal, newVal);
    }

    if (resultMapper.contains(op->getResult(0))) continue;

    Operation *cloned = cloneBuilder.clone(*op, opMapper);
    cloned->setAttr(CVPipeline::kBlockId, cloneBuilder.getI32IntegerAttr(info.nonOwnerBlockId));
    cloned->setAttr(CVPipeline::kArg, cloneBuilder.getI32IntegerAttr(info.argIndex));

    resultMapper.map(op->getResult(0), cloned->getResult(0));
    cloneBuilder.setInsertionPointAfter(cloned);
  }
  return success();
}

// Replaces iter_arg uses in non-owner block with the cloned iter_arg.
// argRemapping maps migrated iter_arg Value -> new extra iter_arg Value.
static LogicalResult replaceIterArgsInBlock(
    SharedArgInfo &info,
    Block *newBlock,
    IRMapping &argRemapping,
    OpBuilder &cloneBuilder)
{
  for (Operation &op : newBlock->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
    if (!blockIdAttr || blockIdAttr.getInt() != info.nonOwnerBlockId) continue;

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      Value operand = op.getOperand(i);
      if (argRemapping.contains(operand)) {
        Value newVal = argRemapping.lookup(operand);
        op.setOperand(i, newVal);
        op.setAttr(CVPipeline::kArg, cloneBuilder.getI32IntegerAttr(info.argIndex));
      }
    }
  }
  return success();
}

// Processes each shared arg: finds insertion point, clones chain, replaces iter_args.
// Collects cloned results for building new yield operands.
//
// `iterArgs` is the list of iter_args of the loop op (forOp.getRegionIterArgs()
// or whileOp.getRegionIterArgs() — same API). `ivOffset` is 1 for scf.for (the
// induction variable sits at block arg 0) and 0 for scf.while (no induction
// variable, iter_args start at 0). `oldBlockArgs` is kept in the signature for
// source compatibility but is no longer used; clonedResults is the sole output.
static LogicalResult processSharedArgsIteration(
    Block *newBlock,
    SmallVector<SharedArgInfo> &sharedArgsInfo,
    const llvm::DenseMap<int, Operation*> &sharedArgToCompOp,
    const llvm::DenseMap<int, llvm::DenseSet<Operation*>> &sharedArgToChainOps,
    ValueRange iterArgs,
    unsigned ivOffset,
    SmallVector<Value> &clonedResults)
{
  unsigned numOriginalIterArgs = iterArgs.size();
  unsigned extraIterArgsBase = ivOffset + numOriginalIterArgs;

  int clonedArgIdx = clonedResults.size();
  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    info.iterArg = iterArgs[argIndex];

    // The migrated iter_arg (original iter_arg moved to new block)
    Value migratedIterArg = newBlock->getArgument(argIndex + ivOffset);
    // The new extra iter_arg added for this shared arg
    unsigned newExtraBlockArgIdx = extraIterArgsBase + info.newArgIndex;
    Value newExtraIterArg = newBlock->getArgument(newExtraBlockArgIdx);

    // Build argRemapping: migratedIterArg -> newExtraIterArg
    IRMapping argRemapping;
    argRemapping.map(migratedIterArg, newExtraIterArg);

    Operation *lastOpInBlock = nullptr;
    for (Operation &op : newBlock->without_terminator()) {
      auto blockIdAttr = op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
      if (blockIdAttr && blockIdAttr.getInt() == info.nonOwnerBlockId) {
        lastOpInBlock = &op;
      }
    }

    OpBuilder cloneBuilder(newBlock, newBlock->end());
    if (lastOpInBlock) {
      cloneBuilder.setInsertionPointAfter(lastOpInBlock);
    }

    IRMapping resultMapper;
    if (failed(cloneChainForBlock(info, sharedArgToCompOp.lookup(argIndex),
                                  sharedArgToChainOps.lookup(argIndex),
                                  newBlock, argRemapping,
                                  cloneBuilder, resultMapper, clonedArgIdx))) {
      continue;
    }

    if (failed(replaceIterArgsInBlock(info, newBlock, argRemapping, cloneBuilder))) {
      continue;
    }

    Value clonedResult = resultMapper.lookup(
        sharedArgToCompOp.lookup(argIndex)->getResult(0));
    clonedResults.push_back(clonedResult);
    clonedArgIdx++;
  }
  return success();
}

// Prepares all shared args data: collects arg->blockId mapping, finds shared args,
// and builds computation info for each shared arg. `loopOp` is the main-loop op
// (scf.for or scf.while); `body` is its loop body (forOp body or whileOp
// after-body). The function dispatches on op type only for getBlockIdsInOrder,
// which has separate forOp/whileOp overloads.
static LogicalResult prepareSharedArgsData(
    Operation *loopOp,
    Block *body,
    SmallVector<SharedArgInfo> &sharedArgsInfo,
    llvm::DenseMap<int, Operation*> &sharedArgToCompOp,
    llvm::DenseMap<int, llvm::DenseSet<Operation*>> &sharedArgToChainOps)
{
  if (!body || !body->mightHaveTerminator()) {
    LDBG("[Error]: loop body is invalid or has no terminator\n");
    return failure();
  }

  // ivOffset: 1 for scf.for (IV at block arg 0; iter_args start at 1), 0 for
  // scf.while (no IV; iter_args start at 0). collectArgIndexToBlockIds indexes
  // its result map by iter_arg index (0-based), so we subtract the offset
  // when storing.
  unsigned ivOffset = isa<scf::ForOp>(loopOp) ? 1 : 0;

  llvm::DenseMap<int, llvm::DenseSet<int>> argIndexToBlockIds;
  if (failed(collectArgIndexToBlockIds(body, ivOffset, argIndexToBlockIds))) {
    return failure();
  }

  SmallVector<int> idsInOrder;
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    idsInOrder = getBlockIdsInOrder(forOp);
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
    idsInOrder = getBlockIdsInOrder(whileOp);
  } else {
    LDBG("[Error]: loopOp is neither scf::ForOp nor scf::WhileOp\n");
    return failure();
  }
  if (failed(findSharedArgs(argIndexToBlockIds, idsInOrder, sharedArgsInfo))) {
    return failure();
  }

  if (sharedArgsInfo.empty()) {
    return success();
  }

  LDBG("[INFO]: Found " << sharedArgsInfo.size() << " shared iter_args to process\n");

  if (failed(buildCompInfoForSharedArgs(loopOp, body, sharedArgsInfo,
                                        sharedArgToCompOp, sharedArgToChainOps))) {
    return failure();
  }

  return success();
}

// Builds new yield op with original operands plus cloned results. Generic over
// the loop op (scf.for or scf.while): uses oldBlock's scf.yield terminator as
// the source of original operands and newBlock (forOp body or whileOp
// after-body) as the destination. Location is taken from newOp.
static LogicalResult buildNewYieldOp(
    Block *oldBlock, Block *newBlock, Operation *newOp,
    const SmallVector<Value> &clonedResults)
{
  auto oldYield = cast<scf::YieldOp>(oldBlock->getTerminator());
  SmallVector<Value> yieldOperands;

  for (unsigned i = 0; i < oldYield.getNumOperands(); ++i) {
    yieldOperands.push_back(oldYield.getOperand(i));
  }
  for (auto &result : clonedResults) {
    yieldOperands.push_back(result);
  }

  OpBuilder builder = OpBuilder::atBlockEnd(newBlock);
  builder.create<scf::YieldOp>(newOp->getLoc(), yieldOperands);
  oldYield.erase();
  return success();
}

// Replaces all uses of the old main-loop op (scf.for or scf.while) with the
// new op's results and erases the old op. Also transfers the
// intraCoreDependentMap entry from the old main-loop op to the new one. The
// map is keyed on Operation* (the main-loop op itself), so this works for
// both scf.for and scf.while.
static LogicalResult replaceForOpAndErase(Operation *oldOp, Operation *newOp,
                                          ControlFlowConditionInfo *info)
{
  if (oldOp->getNumResults() > 0) {
    SmallVector<Value> newResults;
    for (unsigned i = 0; i < oldOp->getNumResults(); ++i) {
      newResults.push_back(newOp->getResult(i));
    }
    oldOp->replaceAllUsesWith(newResults);
  }

  // Transfer intraCoreDependentMap entry from oldOp to newOp.
  if (info) {
    if (info->intraCoreDependentMap.count(oldOp)) {
      info->intraCoreDependentMap[newOp] = info->intraCoreDependentMap[oldOp];
      info->intraCoreDependentMap.erase(oldOp);
    }
  }

  oldOp->erase();
  return success();
}

// Forward declaration: the wrapper that also transfers the
// originalWhileIterArgIndices entry from oldOp to newOp, used by the
// scf.while branch of processSharedIterArgsInLoop.
static LogicalResult replaceForOpAndEraseAndTransferWhileIdx(
    Operation *oldOp, Operation *newOp, ControlFlowConditionInfo *info,
    llvm::DenseMap<scf::WhileOp, SmallVector<unsigned>> &origIdxMap);

// ============================================================
// scf.while support
// ============================================================
//
// The transformations below reuse the generic helpers defined for scf.for
// (collectArgIndexToBlockIds, findCompOpInOwnerBlock, collectChainOps,
// buildCompInfoForSharedArgs, prepareSharedArgsData, processSharedArgsIteration,
// buildNewYieldOp, replaceForOpAndErase) and only add what's truly new for
// scf.while: the scf.while-specific op construction, before+after body
// migration, and the before-region condition extension.
//
// Notes:
//   - scf.while has no induction variable, so before/after block args start at
//     0 with iter_args (processSharedArgsIteration takes ivOffset=0).

// Creates a new scf.while op with extra init args for shared arguments and
// matching extra result types. Empty before/after regions are populated with
// single blocks whose arg types match the new init args (to mirror the count
// carried by init args). The blocks are empty; migration and terminator
// insertion (moveBefore of the original scf.condition + buildNewYieldOp) finish them.
static scf::WhileOp createNewWhileOp(scf::WhileOp whileOp,
                                     const SmallVector<SharedArgInfo> &sharedArgsInfo)
{
  OpBuilder builder(whileOp);

  SmallVector<Value> newInits(whileOp.getInits().begin(), whileOp.getInits().end());
  SmallVector<Type> newResultTypes(whileOp->getResultTypes().begin(),
                                   whileOp->getResultTypes().end());
  // Each non-owner block gets its own extra iter_arg whose init value is the
  // shared-arg's init. The result type grows in lockstep so the loop returns
  // the new iter_args at exit.
  for (const auto &info : sharedArgsInfo) {
    Value init = whileOp.getInits()[info.argIndex];
    newInits.push_back(init);
    newResultTypes.push_back(init.getType());
  }

  scf::WhileOp newWhileOp =
      builder.create<scf::WhileOp>(whileOp.getLoc(), newResultTypes, newInits);

  SmallVector<Type> argTypes;
  argTypes.reserve(newInits.size());
  for (Value v : newInits) {
    argTypes.push_back(v.getType());
  }

  SmallVector<Location> argLocs(newInits.size(), whileOp.getLoc());

  builder.createBlock(&newWhileOp.getBefore(), /*insertBefore=*/{}, argTypes,
                      argLocs);
  builder.createBlock(&newWhileOp.getAfter(), /*insertBefore=*/{}, argTypes,
                      argLocs);

  for (auto &attr : whileOp->getAttrs()) {
    newWhileOp->setAttr(attr.getName(), attr.getValue());
  }
  return newWhileOp;
}

// Main entry point for processing shared iter_args in a single main-loop op
// (scf.for or scf.while). Orchestrates data preparation, new op construction,
// body migration, and cloning. Dispatches on op type at each step so the
// forOp and whileOp pipelines share as much code as possible.
//
// Pipeline (common to both forOp and whileOp):
//   1. prepareSharedArgsData     — collect shared args & build chain info
//   2. createNew*Op              — construct a new loop op with extra iter_args
//   3. migrateBody               — move ops from old block(s) into new block(s)
//   4. processSharedArgsIteration — clone the chain for each non-owner block,
//                                   redirected to the new extra iter_arg
//   5. build terminator(s)       — forOp: scf.yield; whileOp: move original
//                                   scf.condition into new before region + build
//                                   new scf.yield in after region
//   6. replaceForOpAndErase      — splice new op in place of old, transfer any
//                                   intraCoreDependentMap entry to newOp (the
//                                   map is keyed on Operation* now, so the
//                                   same transfer applies to scf.for and
//                                   scf.while). For scf.while, also transfer
//                                   the originalWhileIterArgIndices entry
//                                   (recorded by recordOriginalWhileIterArgs)
//                                   from the old whileOp to the new one so
//                                   the new pipeline in processWhileIterArgs
//                                   can find it.
//
// Differences from forOp to whileOp:
//   - scf.while has no induction variable, so before/after block args start
//     at 0 with iter_args (processSharedArgsIteration takes ivOffset=0).
//   - scf.while has two regions (before + after) instead of one.
//   - scf.while's original scf.condition is preserved unchanged: migrateBody
//     skips terminators, so the old scf.condition is left in the old before
//     block with its cond value's defining op already moved into the new
//     before block and its block-arg uses remapped to the new block's args.
//     We then moveBefore the old scf.condition into the new before block to
//     terminate it without touching the cond logic.
LogicalResult ProcessArgsPass::processSharedIterArgsInLoop(Operation *op,
                                                           ControlFlowConditionInfo *info)
{
  // Dispatch on op type once for the steps that need different inputs
  // (inspect body + ivOffset). Both forOp and whileOp are handled uniformly
  // below the type-specific work.
  Block *inspectBody = nullptr;
  unsigned ivOffset = 0;
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    inspectBody = forOp.getBody();
    ivOffset = 1; // scf.for has the IV at block arg 0; iter_args start at 1.
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    inspectBody = whileOp.getAfterBody();
    ivOffset = 0; // scf.while has no IV; iter_args start at 0.
  } else {
    LDBG("[Error]: op with ssbuffer.main_loop is neither scf::ForOp nor scf::WhileOp\n");
    return failure();
  }

  // 1. Prepare shared args data (dispatches on op type internally for
  //    getBlockIdsInOrder).
  SmallVector<SharedArgInfo> sharedArgsInfo;
  llvm::DenseMap<int, Operation*> sharedArgToCompOp;
  llvm::DenseMap<int, llvm::DenseSet<Operation*>> sharedArgToChainOps;
  if (failed(prepareSharedArgsData(op, inspectBody, sharedArgsInfo,
                                   sharedArgToCompOp, sharedArgToChainOps))) {
    return failure();
  }

  if (sharedArgsInfo.empty()) {
    return success();
  }

  SmallVector<Value> clonedResults;

  // 2-6. Per-op-type pipeline. forOp and whileOp share the call sites
  // (processSharedArgsIteration, replaceForOpAndErase), but the create step
  // (different op factory), the migrate step (1 vs 2 bodies), the iteration
  // target (body + ivOffset), and the final terminator(s) differ.
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    scf::ForOp newForOp = createNewForOp(forOp, sharedArgsInfo);
    Block *oldBlock = forOp.getBody();
    Block *newBlock = newForOp.getBody();
    migrateBody(oldBlock, newBlock);

    if (failed(processSharedArgsIteration(newBlock, sharedArgsInfo,
                                          sharedArgToCompOp, sharedArgToChainOps,
                                          forOp.getRegionIterArgs(), 1,
                                          clonedResults))) {
      return failure();
    }
    if (failed(buildNewYieldOp(oldBlock, newBlock, newForOp, clonedResults))) {
      return failure();
    }
    return replaceForOpAndErase(forOp, newForOp, info);
  }

  // whileOp
  auto whileOp = cast<scf::WhileOp>(op);
  scf::WhileOp newWhileOp = createNewWhileOp(whileOp, sharedArgsInfo);
  migrateBody(whileOp.getBeforeBody(), newWhileOp.getBeforeBody());
  migrateBody(whileOp.getAfterBody(), newWhileOp.getAfterBody());

  if (failed(processSharedArgsIteration(newWhileOp.getAfterBody(), sharedArgsInfo,
                                        sharedArgToCompOp, sharedArgToChainOps,
                                        whileOp.getRegionIterArgs(), 0,
                                        clonedResults))) {
    return failure();
  }

  // Build the new scf.condition in the new before region. The cond value
  // (i1) is preserved from the old scf.condition (its defining op has
  // already been moved into the new before block by migrateBody, and its
  // block-arg uses have been remapped to the new block's args). The
  // forwarded operands must include all of newWhileOp's before-block args,
  // including the extra clone iter_args added by createNewWhileOp — hence
  // we cannot reuse the old scf.condition op verbatim; its forwarded
  // operand count is N while newWhileOp has N+M iter_args. We do NOT OR
  // the cond with any cloned check (the user requested the cond logic stay
  // unchanged).
  auto oldCond = whileOp.getConditionOp();
  Value origCond = oldCond.getCondition();

  OpBuilder beforeBuilder(newWhileOp.getBeforeBody(),
                          newWhileOp.getBeforeBody()->end());
  SmallVector<Value> forwardedValues;
  for (BlockArgument arg : newWhileOp.getBeforeArguments()) {
    forwardedValues.push_back(arg);
  }
  beforeBuilder.create<scf::ConditionOp>(whileOp.getLoc(), origCond,
                                         forwardedValues);
  oldCond.erase();

  // Build the new scf.yield in the new after region.
  if (failed(buildNewYieldOp(whileOp.getAfterBody(), newWhileOp.getAfterBody(),
                             newWhileOp, clonedResults))) {
    return failure();
  }
  return replaceForOpAndEraseAndTransferWhileIdx(whileOp, newWhileOp, info,
                                                 originalWhileIterArgIndices);
}

// Member function wrapper: when we transfer the whileOp to a new op, also
// move the originalWhileIterArgIndices entry from oldOp to newOp so
// processWhileIterArgs (run later in runOnOperation) can find it.
static LogicalResult replaceForOpAndEraseAndTransferWhileIdx(
    Operation *oldOp, Operation *newOp, ControlFlowConditionInfo *info,
    llvm::DenseMap<scf::WhileOp, SmallVector<unsigned>> &origIdxMap)
{
  if (auto oldWhile = dyn_cast<scf::WhileOp>(oldOp)) {
    auto it = origIdxMap.find(oldWhile);
    if (it != origIdxMap.end()) {
      if (auto newWhile = dyn_cast<scf::WhileOp>(newOp)) {
        origIdxMap[newWhile] = it->second;
      }
      origIdxMap.erase(oldWhile);
    }
  }
  return replaceForOpAndErase(oldOp, newOp, info);
}

// Walks module to find for/while ops with ssbuffer.main_loop attribute.
// Processes each main loop to handle shared iter_args. The per-op pipeline
// is shared (see processSharedIterArgsInLoop); this function only handles
// the type-agnostic walk + dispatch into the unified entry point.
LogicalResult ProcessArgsPass::processSharedIterArgs(ModuleOp module)
{
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr(CVPipeline::kMainLoop)) {
      return WalkResult::advance();
    }
    if (failed(processSharedIterArgsInLoop(op, info))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }
  return success();
}

// ============================================================
// Per-block update chain for scf.while iter_args used in scf.condition
// ============================================================
//
// For each scf.while op with main_loop attribute, we want every block_id's
// region of the do block to maintain its own iter_arg value (one new arg per
// block per original iter_arg used in scf.condition). To do this we:
//   1. At the start of ProcessArgs, snapshot the original iter_args so we
//      can later tell which iter_args were already in scf.condition.
//   2. After processSharedIterArgs, for every scf.while op with main_loop,
//      find iter_args in scf.condition, locate the "update chain" in the
//      after body (the ops that ultimately produce the value yielded for
//      that iter_arg), and clone that chain at the end of each block_id's
//      run of consecutive ops. Each clone is annotated with
//      `ssbuffer.while_arg = <original index>` and `ssbuffer.block_id = <target>`.
//   3. Build a new scf.while with one extra iter_arg per (block_id, original
//      iter_arg), extend scf.condition and scf.yield accordingly, and remap
//      each block's update-chain input to its own new iter_arg.
//   4. Record whileBlockArgMap[whileop][block_id][new_arg_idx] = old_arg_idx
//      in info for downstream passes.

// Snapshot the original iter_args of every scf.while op with main_loop attr.
// Must run before any other processing so we can later identify which
// iter_args were referenced by scf.condition in the input.
void ProcessArgsPass::recordOriginalWhileIterArgs(ModuleOp module)
{
  module.walk([&](scf::WhileOp whileOp) {
    if (!whileOp->hasAttr(CVPipeline::kMainLoop)) {
      return;
    }
    SmallVector<unsigned> indices;
    for (unsigned i = 0; i < whileOp.getNumOperands(); ++i) {
      // For scf.while, getNumOperands() equals the number of init args, which
      // equals the number of iter_args. Index i is the iter_arg index.
      indices.push_back(i);
    }
    originalWhileIterArgIndices[whileOp] = indices;
  });
}

// Returns the set of original iter_arg indices used in scf.condition for
// `whileOp`. An original iter_arg index is used in scf.condition if its
// corresponding block argument appears among the forwarded operands of the
// condition op (operand 0 is the cond value, operands 1..N are forwarded
// iter_args). Restrict to originalIndices; iter_args added by
// processSharedIterArgs (past originalIndices.size()) are ignored.
static llvm::DenseSet<unsigned> collectConditionUsedIterArgIndices(
    scf::WhileOp whileOp, const SmallVector<unsigned> &originalIndices)
{
  llvm::DenseSet<unsigned> used;
  auto cond = whileOp.getConditionOp();
  for (OpOperand &operand : cond->getOpOperands()) {
    if (operand.getOperandNumber() == 0) {
      continue; // cond value
    }
    Value v = operand.get();
    auto beforeArg = dyn_cast<BlockArgument>(v);
    if (!beforeArg) continue;
    unsigned idx = beforeArg.getArgNumber();
    if (idx < originalIndices.size()) {
      used.insert(idx);
    }
  }
  return used;
}

// Locates the compOp in the after body that produces the value yielded for
// iter_arg at `argIndex` to scf.yield. The compOp is the defining op of the
// yielded value (the top of the update chain). Returns nullptr on failure.
static Operation *findYieldCompOpForWhile(scf::WhileOp whileOp, unsigned argIndex)
{
  Block *body = whileOp.getAfterBody();
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
  if (argIndex >= yieldOp.getNumOperands()) {
    return nullptr;
  }
  Value yieldArg = yieldOp.getOperand(argIndex);
  return yieldArg.getDefiningOp();
}

// Returns the last op in `body` whose `ssbuffer.block_id` matches `blockId`.
// The body is the after region of an scf.while. Searches in textual order so
// it returns the latest op with that block_id (which is what the user asked
// for: end of each block's run of consecutive ops).
static Operation *findLastOpWithBlockId(Block *body, int blockId)
{
  Operation *last = nullptr;
  for (Operation &op : body->without_terminator()) {
    auto attr = op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
    if (attr && attr.getInt() == blockId) {
      last = &op;
    }
  }
  return last;
}

// For a single iter_arg (used in scf.condition) and a single target block_id,
// build a clone of the update chain after the last op with that block_id in
// the after body. The clone is annotated with `ssbuffer.while_arg` =
// `originalArgIndex` and `ssbuffer.block_id` = `blockId`. Operands that
// refer to the original iter_arg (block arg at `originalArgIndex`) are
// remapped to the new iter_arg (block arg at `newArgIndex`); operands that
// refer to chain-internal values are remapped to the corresponding cloned
// values. Returns the cloned compOp result (the value to be yielded for
// `newArgIndex`).
static Value cloneUpdateChainForWhileBlock(
    scf::WhileOp whileOp, Block *body, Operation *compOp,
    int blockId, unsigned originalArgIndex, unsigned newArgIndex,
    const llvm::DenseSet<Operation *> &chainOps)
{
  if (!compOp || chainOps.empty()) {
    return Value();
  }

  SmallVector<Operation *> sortedChain(chainOps.begin(), chainOps.end());
  if (failed(topologicalSort(sortedChain))) {
    return Value();
  }

  // Find insertion point: after the last op with this block_id.
  Operation *lastOp = findLastOpWithBlockId(body, blockId);
  OpBuilder builder(body, body->end());
  if (lastOp) {
    builder.setInsertionPointAfter(lastOp);
  } else {
    // No op with this block_id; append at end (before terminator).
    builder.setInsertionPoint(body->getTerminator());
  }

  IRMapping resultMapper;
  for (Operation *op : sortedChain) {
    IRMapping opMapper;
    for (OpOperand &operand : op->getOpOperands()) {
      Value oldVal = operand.get();
      Value newVal = oldVal;
      // The original iter_arg (block arg) should be remapped to the new
      // iter_arg for this block.
      auto blockArg = dyn_cast<BlockArgument>(oldVal);
      if (blockArg && blockArg.getOwner() == body &&
          (unsigned)blockArg.getArgNumber() == originalArgIndex) {
        newVal = body->getArgument(newArgIndex);
      } else if (resultMapper.contains(oldVal)) {
        // Operand is an intermediate from earlier in the cloned chain.
        newVal = resultMapper.lookup(oldVal);
      }
      opMapper.map(oldVal, newVal);
    }

    if (resultMapper.contains(op->getResult(0))) continue;

    Operation *cloned = builder.clone(*op, opMapper);
    cloned->setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(blockId));
    cloned->setAttr(CVPipeline::kWhileArg, builder.getI32IntegerAttr(originalArgIndex));
    resultMapper.map(op->getResult(0), cloned->getResult(0));
    builder.setInsertionPointAfter(cloned);
  }

  return resultMapper.lookup(compOp->getResult(0));
}

// Process a single scf.while op: for each original iter_arg used in
// scf.condition, clone its update chain into the end of each block_id's run
// of consecutive ops, build a new scf.while with one extra iter_arg per
// (block_id, original iter_arg), redirect each block's update chain input to
// its own new iter_arg, extend scf.condition + scf.yield, and record
// whileBlockArgMap[whileop][block_id][new_arg_idx] = old_arg_idx.
LogicalResult ProcessArgsPass::processWhileIterArgsInWhileOp(
    scf::WhileOp whileOp, ControlFlowConditionInfo *info)
{
  // Find original iter_arg indices snapshot; bail out if missing (the pass
  // didn't snapshot, e.g. caller forgot to call recordOriginalWhileIterArgs).
  auto it = originalWhileIterArgIndices.find(whileOp);
  if (it == originalWhileIterArgIndices.end()) {
    return success();
  }
  const SmallVector<unsigned> &originalIndices = it->second;

  llvm::DenseSet<unsigned> condUsed =
      collectConditionUsedIterArgIndices(whileOp, originalIndices);
  if (condUsed.empty()) {
    return success();
  }

  // Find all distinct block_ids in the after body, in order of appearance.
  SmallVector<int> blockIdsInOrder = getBlockIdsInOrder(whileOp);
  if (blockIdsInOrder.empty()) {
    return success();
  }

  Block *body = whileOp.getAfterBody();

  // For each (block_id, originalArgIndex), we will add one new iter_arg.
  // `clonedPerBlock[block_id]` stores cloned compOp results in the order of
  // originalIndices, so we can later extend scf.yield.
  llvm::DenseMap<int, SmallVector<Value>> clonedPerBlock;
  SmallVector<std::tuple<int, unsigned, unsigned>> newArgDescriptors; // (block_id, new_arg_idx, original_arg_idx)

  unsigned nextNewArgIdx = whileOp.getNumOperands();
  // Map<block_id, count of new args added so far for this block>
  llvm::DenseMap<int, unsigned> newArgCountPerBlock;

  // Compute the compOp and its def-chain once, here, BEFORE migrateBody moves
  // the ops into the new while op's after body. The Operation* pointers stay
  // valid across migrateBody (moveBefore relinks rather than destroys), and
  // collectChainOps must be scoped to the *old* whileOp because that is where
  // the ops currently live. Re-querying after migrateBody is unsafe (the old
  // body is stale) and was the source of the IRMapping::lookup crash.
  llvm::DenseMap<unsigned, Operation *> origIdxToCompOp;
  llvm::DenseMap<unsigned, llvm::DenseSet<Operation *>> origIdxToChainOps;

  for (unsigned origIdx : originalIndices) {
    if (!condUsed.contains(origIdx)) continue;

    Operation *compOp = findYieldCompOpForWhile(whileOp, origIdx);
    if (!compOp) {
      LDBG("[WARN]: no compOp for while iter_arg idx=" << origIdx);
      continue;
    }

    // Collect chain ops (the def-chain rooted at compOp, scoped to the
    // whileOp's body).
    llvm::DenseSet<Operation *> chainOps;
    collectChainOps(whileOp, compOp, chainOps);
    origIdxToCompOp[origIdx] = compOp;
    origIdxToChainOps[origIdx] = chainOps;

    for (int blockId : blockIdsInOrder) {
      unsigned newArgIdx = nextNewArgIdx++;
      clonedPerBlock[blockId].push_back(Value()); // placeholder, filled below
      newArgDescriptors.push_back({blockId, newArgIdx, origIdx});
      newArgCountPerBlock[blockId]++;
    }
  }

  if (newArgDescriptors.empty()) {
    return success();
  }

  // Build the new scf.while op with one extra init arg per descriptor.
  // Each new init arg's init value is the same as the original iter_arg
  // (matching the user's example: %arg19, %arg20, %arg21 all init from %c0_i32).
  OpBuilder builder(whileOp);
  SmallVector<Value> newInits(whileOp.getInits().begin(), whileOp.getInits().end());
  SmallVector<Type> newResultTypes(whileOp->getResultTypes().begin(),
                                   whileOp->getResultTypes().end());
  for (auto &desc : newArgDescriptors) {
    int blockId;
    unsigned newArgIdx, origIdx;
    std::tie(blockId, newArgIdx, origIdx) = desc;
    Value initVal = whileOp.getInits()[origIdx];
    newInits.push_back(initVal);
    newResultTypes.push_back(initVal.getType());
  }

  scf::WhileOp newWhileOp =
      builder.create<scf::WhileOp>(whileOp.getLoc(), newResultTypes, newInits);

  // Add before and after regions with matching arg types.
  SmallVector<Type> argTypes;
  argTypes.reserve(newInits.size());
  for (Value v : newInits) {
    argTypes.push_back(v.getType());
  }
  SmallVector<Location> argLocs(newInits.size(), whileOp.getLoc());
  builder.createBlock(&newWhileOp.getBefore(), {}, argTypes, argLocs);
  builder.createBlock(&newWhileOp.getAfter(), {}, argTypes, argLocs);

  // Copy attributes from the old whileOp (incl. ssbuffer.main_loop).
  for (auto &attr : whileOp->getAttrs()) {
    newWhileOp->setAttr(attr.getName(), attr.getValue());
  }

  // Migrate body: remap old before/after block args to new ones, move ops.
  Block *oldBefore = whileOp.getBeforeBody();
  Block *newBefore = newWhileOp.getBeforeBody();
  Block *oldAfter = whileOp.getAfterBody();
  Block *newAfter = newWhileOp.getAfterBody();
  migrateBody(oldBefore, newBefore);
  migrateBody(oldAfter, newAfter);

  // Now do the actual cloning inside newAfter for each (block_id, original
  // arg index) using newAfter's block args for remapping.
  unsigned descIdx = 0;
  // Group descriptors by block_id and original arg index so that we clone
  // per (origIdx) once and visit each block.
  // First, build a map: (block_id, origIdx) -> newArgIdx.
  llvm::DenseMap<std::pair<int, unsigned>, unsigned> blockOrigToNewArg;
  for (auto &desc : newArgDescriptors) {
    int blockId;
    unsigned newArgIdx, origIdx;
    std::tie(blockId, newArgIdx, origIdx) = desc;
    blockOrigToNewArg[{blockId, origIdx}] = newArgIdx;
  }

  for (unsigned origIdx : originalIndices) {
    if (!condUsed.contains(origIdx)) continue;

    // Reuse the compOp/chainOps computed before migrateBody. The ops now live
    // in newAfter (moved by migrateBody), so their operands already reference
    // newAfter's block args, and cloneUpdateChainForWhileBlock can clone them
    // in place scoped to newAfter.
    Operation *compOp = origIdxToCompOp.lookup(origIdx);
    if (!compOp) continue;
    const llvm::DenseSet<Operation *> &chainOps = origIdxToChainOps[origIdx];

    for (int blockId : blockIdsInOrder) {
      unsigned newArgIdx = blockOrigToNewArg.lookup({blockId, origIdx});
      Value cloned = cloneUpdateChainForWhileBlock(
          newWhileOp, newAfter, compOp, blockId, origIdx, newArgIdx, chainOps);
      clonedPerBlock[blockId][descIdx] = cloned;
    }
    descIdx++;
  }

  // Build new scf.condition in the new before region. Forwarded values are
  // the new before-block args (matching newInits count). Cond value is
  // preserved from the old scf.condition.
  auto oldCond = whileOp.getConditionOp();
  Value origCond = oldCond.getCondition();
  OpBuilder beforeBuilder(newBefore, newBefore->end());
  SmallVector<Value> forwarded;
  for (BlockArgument arg : newWhileOp.getBeforeArguments()) {
    forwarded.push_back(arg);
  }
  beforeBuilder.create<scf::ConditionOp>(whileOp.getLoc(), origCond, forwarded);
  oldCond.erase();

  // Build new scf.yield in the new after region. Yield operands are: the
  // original yield operands (which are the values for the original iter_args
  // used in scf.condition; they come from the migrated ops), then the cloned
  // results per (block_id, original arg index), in the order of
  // newArgDescriptors.
  auto oldYield = cast<scf::YieldOp>(oldAfter->getTerminator());
  SmallVector<Value> yieldOperands;
  for (unsigned i = 0; i < oldYield.getNumOperands(); ++i) {
    yieldOperands.push_back(oldYield.getOperand(i));
  }
  // Append cloned results in the order of newArgDescriptors, which is
  // (block_id_1, origIdx_1), (block_id_1, origIdx_2), ..., (block_id_2, ...).
  for (auto &desc : newArgDescriptors) {
    int blockId;
    unsigned newArgIdx, origIdx;
    std::tie(blockId, newArgIdx, origIdx) = desc;
    auto &vec = clonedPerBlock[blockId];
    // Find which position in vec corresponds to origIdx.
    // We re-derive it: vec is in order of originalIndices iteration, and we
    // only push for condUsed iter_args in the order they appear in
    // originalIndices. Use a counter.
    // Easier: re-iterate and build a map origIdx -> position in vec.
    // But we don't have that here; build it on the fly.
    // Track by re-iterating originalIndices and finding the condUsed rank.
    unsigned pos = 0;
    for (unsigned oi : originalIndices) {
      if (!condUsed.contains(oi)) continue;
      if (oi == origIdx) break;
      pos++;
    }
    yieldOperands.push_back(vec[pos]);
  }
  OpBuilder afterBuilder(newAfter, newAfter->end());
  afterBuilder.create<scf::YieldOp>(whileOp.getLoc(), yieldOperands);
  oldYield.erase();

  // Record whileBlockArgMap for downstream passes (info, when set) and
  // always into the pass's local copy (localWhileBlockArgMap), so the map
  // is observable when the pass is run standalone (--process-args) without
  // going through AddControlFlowConditionPass, where `info` may be null.
  for (auto &desc : newArgDescriptors) {
    int blockId;
    unsigned newArgIdx, origIdx;
    std::tie(blockId, newArgIdx, origIdx) = desc;
    localWhileBlockArgMap[newWhileOp][blockId][newArgIdx] = (int)origIdx;
    if (info) {
      info->whileBlockArgMap[newWhileOp][blockId][newArgIdx] = (int)origIdx;
    }
  }

  // Replace all uses of the old whileOp with the new one, transfer
  // intraCoreDependentMap entry, then erase the old whileOp.
  if (whileOp->getNumResults() > 0) {
    SmallVector<Value> newResults;
    for (unsigned i = 0; i < whileOp->getNumResults(); ++i) {
      newResults.push_back(newWhileOp->getResult(i));
    }
    whileOp->replaceAllUsesWith(newResults);
  }
  if (info) {
    if (info->intraCoreDependentMap.count(whileOp)) {
      info->intraCoreDependentMap[newWhileOp] = info->intraCoreDependentMap[whileOp];
      info->intraCoreDependentMap.erase(whileOp);
    }
  }
  whileOp->erase();

  return success();
}

LogicalResult ProcessArgsPass::processWhileIterArgs(ModuleOp module)
{
  // Walk collects all while ops with main_loop first, then process them.
  // We must NOT mutate the IR during the walk (we replace the old whileOp
  // with a new one in processWhileIterArgsInWhileOp), so collect first.
  SmallVector<scf::WhileOp> worklist;
  module.walk([&](scf::WhileOp whileOp) {
    if (whileOp->hasAttr(CVPipeline::kMainLoop)) {
      worklist.push_back(whileOp);
    }
  });
  for (scf::WhileOp whileOp : worklist) {
    if (failed(processWhileIterArgsInWhileOp(whileOp, info))) {
      return failure();
    }
  }

  // Dump whileBlockArgMap for debugging — shows the (whileop -> block_id ->
  // (new_arg_idx -> old_arg_idx)) mapping. Each whileOp's entry is keyed on
  // the NEW while op built by processWhileIterArgsInWhileOp. Uses the
  // pass-local map (always populated) so this is observable even when the
  // pass is run standalone (--process-args) without going through
  // AddControlFlowConditionPass, where `info` may be null.
  LDBG("[INFO]: whileBlockArgMap contents (new_whileop -> block_id -> "
       "(new_arg_idx -> old_arg_idx)):\n");
  for (auto &whileEntry : localWhileBlockArgMap) {
    scf::WhileOp w = whileEntry.first;
    LDBG("  whileOp @" << w.getLoc() << ":\n");
    for (auto &blockEntry : whileEntry.second) {
      int blockId = blockEntry.first;
      for (auto &argEntry : blockEntry.second) {
        int newArgIdx = argEntry.first;
        int oldArgIdx = argEntry.second;
        LDBG("    block_id=" << blockId
                            << " new_arg_idx=" << newArgIdx
                            << " -> old_arg_idx=" << oldArgIdx << "\n");
      }
    }
  }

  return success();
}

void ProcessArgsPass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("before processArgs:\n" << module << "\n");

  // 1. Snapshot original iter_args of every scf.while op with main_loop
  //    before any other processing happens.
  recordOriginalWhileIterArgs(module);

  // 2. Existing pipeline: process shared iter_args (adds per-block clones
  //    for args shared across block_ids).
  if (failed(processSharedIterArgs(module))) {
    signalPassFailure();
    return;
  }

  // 3. New pipeline: for each scf.while op with main_loop, clone the update
  //    chain of every iter_arg used in scf.condition into the end of each
  //    block_id's run of consecutive ops in the do body, and add a new
  //    iter_arg per (block_id, original arg) pair. Recorded in
  //    info.whileBlockArgMap for downstream passes.
  if (failed(processWhileIterArgs(module))) {
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
