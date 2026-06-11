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

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow/AnalyzeArgs.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinTypes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"

static constexpr const char *DEBUG_TYPE = "analyze-args-in-forOps";
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

namespace {

static constexpr llvm::StringLiteral containedFunc[] {
  "chunk_gated_delta_rule_bwd_kernel_dhu_k128_blockdim128",
};

static LogicalResult isInterceptedModule(ModuleOp module)
{
  bool intercepted = false;

  module.walk([&](func::FuncOp funcOp) -> WalkResult {
    if (!llvm::is_contained(containedFunc, funcOp.getSymName())) {
      return WalkResult::advance();
    }
    intercepted = true;
    return WalkResult::interrupt();
  });

  if (!intercepted) {
    return success();
  }

  return failure();
}

// Pulled in here as a local alias for concise call sites within this file.
using CVPipeline::getOpBlockId;
using CVPipeline::getTensorIterArgIndex;

// Collect block info for all tensor-type iter_args in the forOp body
static llvm::DenseMap<unsigned, TensorArgBlockInfo>
collectTensorArgBlockInfo(scf::ForOp forOp)
{
  llvm::DenseMap<unsigned, TensorArgBlockInfo> result;

  Block *body = forOp.getBody();
  if (!body) {
    return result;
  }

  for (Operation &op : body->without_terminator()) {
    auto blockIdOpt = CVPipeline::getOpBlockId(&op);
    if (!blockIdOpt) {
      continue;
    }
    int blockId = static_cast<int>(*blockIdOpt);

    for (OpOperand &operand : op.getOpOperands()) {
      int argIdx = CVPipeline::getTensorIterArgIndex(operand.get(), forOp);
      if (argIdx >= 0) {
        auto &info = result[argIdx];
        if (info.firstBlockId < 0) {
          info.firstBlockId = blockId;
        }
      }
    }
  }

  return result;
}

// Check if the first use block_id differs from the block_id of yield's defining op.
// Because checkTensorArgsUseAfterUpdate guarantees that every use of a tensor
// iter_arg happens before its update, a "different block_id" between use and
// update is the only remaining signal that matters.
static LogicalResult checkUseUpdateMismatch(scf::ForOp forOp,
                                            const llvm::DenseMap<unsigned, TensorArgBlockInfo> &argBlockInfo)
{
  Block *body = forOp.getBody();
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());

  for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
    if (!isa<RankedTensorType>(forOp.getRegionIterArgs()[i].getType())) {
      continue;
    }

    auto it = argBlockInfo.find(i);
    if (it == argBlockInfo.end()) {
      continue;
    }

    Operation *defOp = yieldOp.getOperand(i).getDefiningOp();
    if (!defOp) {
      continue;
    }

    auto defBlockIdOpt = CVPipeline::getOpBlockId(defOp);
    if (!defBlockIdOpt) {
      continue;
    }

    if (it->second.firstBlockId != *defBlockIdOpt) {
      LDBG("[INFO]: Found tensor iter_arg using and updating in different block_ids!");
      return failure();
    }
  }
  return success();
}

// Main check function: a tensor iter_arg whose first use block_id differs from
// the block_id of its update op makes the main_loop non-serial.
static LogicalResult hasTensorArgInDifferentBlockIds(scf::ForOp forOp)
{
  auto argBlockInfo = collectTensorArgBlockInfo(forOp);
  if (failed(checkUseUpdateMismatch(forOp, argBlockInfo))) {
    return failure();
  }
  return success();
}

// For a single main_loop forOp, find every block_id (within the forOp body)
// that contains a hivm.hir.sync_block_wait op -- these are the blocks that
// actually receive cross-core data and are the "real" block nodes for the
// VECTOR/CUBE dataflow graph.
static void collectBlockNodesInMainLoop(scf::ForOp forOp, int mainloopId,
                                        const std::string &belongingScope,
                                        llvm::SmallVectorImpl<BlockNodeInfo> &blockNodes)
{
  // Each block_id should only contribute one node per (mainloopId, scope).
  llvm::DenseSet<int> seenBlockIds;
  forOp.walk([&](hivm::SyncBlockWaitOp syncWaitOp) {
    auto blockIdOpt = getOpBlockId(syncWaitOp);
    if (!blockIdOpt) {
      return;
    }
    int blockId = static_cast<int>(*blockIdOpt);
    if (seenBlockIds.contains(blockId)) {
      return;
    }
    seenBlockIds.insert(blockId);

    BlockNodeInfo node;
    node.mainloopId = mainloopId;
    node.belongingScope = belongingScope;
    node.blockId = blockId;
    blockNodes.push_back(std::move(node));
  });
}

// Build the VECTOR/CUBE dataflow graph for every main_loop forOp in the
// module. Each main_loop's block nodes are stored in its MainLoopData.
//
// A block is treated as a "real" block node only when it contains a
// hivm.hir.sync_block_wait op (i.e. it is on the receiving side of an
// inter-core data transfer). Blocks that do not participate in cross-core
// communication are intentionally excluded.
static void buildDataFlowGraph(ModuleOp module,
                               llvm::SmallVectorImpl<MainLoopData> &mainLoops)
{
  CVPipeline::walkMainLoopForOps(module, [&](scf::ForOp forOp, int mainloopId) -> WalkResult {
    MainLoopData data;
    data.forOp = forOp;
    data.mainloopId = mainloopId;
    data.scope = CVPipeline::getEnclosingScope(forOp);

    collectBlockNodesInMainLoop(forOp, data.mainloopId, data.scope,
                                data.blockNodes);

    LDBG("[INFO]: main_loop id=" << data.mainloopId << ", scope=" << data.scope
                                 << ", node count=" << data.blockNodes.size()
                                );
    mainLoops.push_back(std::move(data));
    return WalkResult::advance();
  });
}

// Walk every annotation.mark op in the module and build a map
// `markedValue -> hivm.tightly_coupled_buffer id`. The id is the
// cross-scope equivalence key: same id on the VECTOR side and on the CUBE
// side means the two memref.alloc values refer to the same physical buffer.
static void buildMarkValueToTcbIdMap(ModuleOp module,
                                     llvm::DenseMap<Value, int> &valueToTcbId)
{
  module.walk([&](annotation::MarkOp markOp) {
    auto tcbAttr =
        markOp->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>(
            CVPipeline::kTightlyCoupledBuffer);
    if (!tcbAttr) {
      return;
    }
    auto id = tcbAttr.getId();
    if (!id) {
      return;
    }
    valueToTcbId[markOp.getOperand(0)] = id.value();
  });
}

// Build a blockId -> BlockNodeInfo* lookup table for the given block nodes.
// Used to resolve a walked op's `ssbuffer.block_id` back to its node in
// O(1) inside the per-op callback.
static llvm::DenseMap<int, BlockNodeInfo *>
buildBlockIdToNodeMap(llvm::SmallVectorImpl<BlockNodeInfo> &blockNodes)
{
  llvm::DenseMap<int, BlockNodeInfo *> blockIdToNode;
  for (BlockNodeInfo &node : blockNodes) {
    blockIdToNode[node.blockId] = &node;
  }
  return blockIdToNode;
}

// Append targetNode to srcNode->outNodes, deduping. A block may emit
// multiple copies / fixpipes to the same peer node, so a single dedup
// pass avoids duplicate edges in the adjacency list.
static void addUniqueOutNode(BlockNodeInfo *srcNode, BlockNodeInfo *targetNode)
{
  if (!llvm::is_contained(srcNode->outNodes, targetNode)) {
    srcNode->outNodes.push_back(targetNode);
  }
}

// Build tcb_id -> BlockNode* reverse map for a forOp's block nodes by
// walking every op, finding the underlying memref.alloc for each operand,
// and resolving its tcb_id. Used in both directions:
//   - CUBE side: alloc is consumed directly (traceCasts=false)
//   - VECTOR side: alloc is consumed through casts (traceCasts=true)
static void buildTcbConsumerMap(
    scf::ForOp forOp,
    llvm::SmallVectorImpl<BlockNodeInfo> &blockNodes,
    const llvm::DenseMap<Value, int> &valueToTcbId,
    bool traceCasts,
    llvm::DenseMap<int, BlockNodeInfo *> &tcbToNode)
{
  auto blockIdToNode = buildBlockIdToNodeMap(blockNodes);

  forOp.walk([&](Operation *op) {
    auto blockIdOpt = getOpBlockId(op);
    if (!blockIdOpt) {
      return;
    }
    auto nodeIt = blockIdToNode.find(static_cast<int>(*blockIdOpt));
    if (nodeIt == blockIdToNode.end()) {
      return;
    }
    BlockNodeInfo *node = nodeIt->second;

    for (Value operand : op->getOperands()) {
      memref::AllocOp allocOp = CVPipeline::findAlloc(operand, traceCasts);
      if (!allocOp) {
        continue;
      }
      auto tcbIt = valueToTcbId.find(allocOp.getResult());
      if (tcbIt != valueToTcbId.end()) {
        tcbToNode[tcbIt->second] = node;
      }
    }
  });
}

// Populate outNodes on the src side by walking a specific inter-core
// transfer op type and matching its dst's tcb_id against tcbToTargetNode.
//   - VECTOR -> CUBE:    OpType = hivm::CopyOp,    tcbToTargetNode = tcbToCubeNode
//   - CUBE   -> VECTOR:  OpType = hivm::FixpipeOp, tcbToTargetNode = tcbToVectorNode
template <typename OpType>
static void populateOutNodesViaTransferOp(
    scf::ForOp forOp,
    llvm::SmallVectorImpl<BlockNodeInfo> &srcBlockNodes,
    const llvm::DenseMap<Value, int> &valueToTcbId,
    const llvm::DenseMap<int, BlockNodeInfo *> &tcbToTargetNode)
{
  auto blockIdToNode = buildBlockIdToNodeMap(srcBlockNodes);

  forOp.walk([&](OpType op) {
    auto blockIdOpt = getOpBlockId(op);
    if (!blockIdOpt) {
      return;
    }
    auto nodeIt = blockIdToNode.find(static_cast<int>(*blockIdOpt));
    if (nodeIt == blockIdToNode.end()) {
      return;
    }
    BlockNodeInfo *srcNode = nodeIt->second;

    Value dst = op.getDst();
    auto tcbIt = valueToTcbId.find(dst);
    if (tcbIt == valueToTcbId.end()) {
      return;
    }

    auto targetIt = tcbToTargetNode.find(tcbIt->second);
    if (targetIt == tcbToTargetNode.end()) {
      return;
    }
    addUniqueOutNode(srcNode, targetIt->second);
  });
}

// Forward DFS: can `start` reach any other-scope node in `scopeSet` by
// walking the dataflow outNodes (which may pass through nodes of other
// scopes)? A node whose outNodes all terminate inside other-scope land
// returns false -- it is an outDegreeZero candidate.
static bool canReachOtherScopeNode(
    BlockNodeInfo *start,
    const llvm::DenseSet<BlockNodeInfo *> &scopeSet)
{
  llvm::DenseSet<BlockNodeInfo *> visited;
  llvm::SmallVector<BlockNodeInfo *, 16> stack;
  stack.push_back(start);
  visited.insert(start);
  while (!stack.empty()) {
    BlockNodeInfo *cur = stack.pop_back_val();
    for (BlockNodeInfo *next : cur->outNodes) {
      if (!visited.insert(next).second) {
        continue;
      }
      if (scopeSet.contains(next)) {
        return true;
      }
      stack.push_back(next);
    }
  }
  return false;
}

// Backward DFS: can any other-scope node in `scopeSet` reach `start` by
// walking the reverse dataflow (which may pass through nodes of other
// scopes)? A node whose inNodes all originate inside other-scope land
// returns false -- it is an inDegreeZero candidate.
static bool isReachableFromOtherScopeNode(
    BlockNodeInfo *start,
    const llvm::DenseSet<BlockNodeInfo *> &scopeSet,
    const llvm::DenseMap<BlockNodeInfo *,
                         llvm::SmallVector<BlockNodeInfo *, 4>> &reverseGraph)
{
  llvm::DenseSet<BlockNodeInfo *> visited;
  llvm::SmallVector<BlockNodeInfo *, 16> stack;
  stack.push_back(start);
  visited.insert(start);
  while (!stack.empty()) {
    BlockNodeInfo *cur = stack.pop_back_val();
    auto revIt = reverseGraph.find(cur);
    if (revIt == reverseGraph.end()) {
      continue;
    }
    for (BlockNodeInfo *prev : revIt->second) {
      if (!visited.insert(prev).second) {
        continue;
      }
      if (scopeSet.contains(prev)) {
        return true;
      }
      stack.push_back(prev);
    }
  }
  return false;
}

// For every main_loop's block nodes whose scope matches `scope`, find the
// "boundary" nodes in the scope-induced reachability:
//   - inDegreeZero:  no other same-scope node can reach this one
//   - outDegreeZero: this node cannot reach any other same-scope node
//
// Reachability is computed on the full dataflow graph, so a chain like
// <scope> -> <other scope> -> <scope> still counts as same-scope
// reachability. A node with all in-edges from other-scope nodes whose own
// in-chains never reach this scope is treated as if it has no in-edges;
// the symmetric rule applies to outDegreeZero.
static DegreeInfo findDegreeZeroNodesInScope(
    llvm::ArrayRef<MainLoopData> mainLoops,
    llvm::StringRef scope)
{
  DegreeInfo result;

  // Collect all same-scope nodes across mainloops.
  llvm::SmallVector<BlockNodeInfo *, 8> scopeNodes;
  llvm::DenseSet<BlockNodeInfo *> scopeSet;
  for (const MainLoopData &data : mainLoops) {
    if (data.scope != scope) {
      continue;
    }
    for (const BlockNodeInfo &node : data.blockNodes) {
      BlockNodeInfo *p = const_cast<BlockNodeInfo *>(&node);
      scopeNodes.push_back(p);
      scopeSet.insert(p);
    }
  }

  if (scopeNodes.empty()) {
    return result;
  }

  // Build reverse edges (target -> list of sources) once for the backward DFS.
  llvm::DenseMap<BlockNodeInfo *, llvm::SmallVector<BlockNodeInfo *, 4>>
      reverseGraph;
  for (const MainLoopData &data : mainLoops) {
    for (const BlockNodeInfo &node : data.blockNodes) {
      for (BlockNodeInfo *out : node.outNodes) {
        reverseGraph[out].push_back(const_cast<BlockNodeInfo *>(&node));
      }
    }
  }

  for (BlockNodeInfo *node : scopeNodes) {
    if (!canReachOtherScopeNode(node, scopeSet)) {
      result.outDegreeZero.push_back(node);
    }
    if (!isReachableFromOtherScopeNode(node, scopeSet, reverseGraph)) {
      result.inDegreeZero.push_back(node);
    }
  }

  return result;
}

// Log a DegreeInfo result in the standard "[INFO]: <scope> <dir>-degree-0"
// format. Used by analyzeMainLoopDataflow to print both VECTOR and CUBE
// results in a consistent way.
static void logDegreeInfo(llvm::StringRef scope, const DegreeInfo &info)
{
  LDBG("[INFO]: " << scope << " in-degree-0 (entry) nodes: "
                  << info.inDegreeZero.size());
  for (BlockNodeInfo *n : info.inDegreeZero) {
    LDBG("  mainloopId=" << n->mainloopId
                         << ", blockId=" << n->blockId);
  }
  LDBG("[INFO]: " << scope << " out-degree-0 (exit) nodes: "
                  << info.outDegreeZero.size());
  for (BlockNodeInfo *n : info.outDegreeZero) {
    LDBG("  mainloopId=" << n->mainloopId
                         << ", blockId=" << n->blockId);
  }
}

// Walk the forOp body once and build a block_id -> iter_arg usage map.
// An op without a `ssbuffer.block_id` attribute is skipped (it is either
// outside the ssbuffer block system or a non-block op).
static llvm::DenseMap<int, BlockIterArgUsage>
collectBlockIterArgUsage(scf::ForOp forOp)
{
  llvm::DenseMap<int, BlockIterArgUsage> result;
  if (!forOp) {
    return result;
  }
  Block *body = forOp.getBody();
  if (!body) {
    return result;
  }
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());

  for (Operation &op : body->without_terminator()) {
    auto blockIdOpt = getOpBlockId(&op);
    if (!blockIdOpt) {
      continue;
    }
    int blockId = static_cast<int>(*blockIdOpt);
    BlockIterArgUsage &usage = result[blockId];

    // Which iter_args does this op consume?
    for (OpOperand &operand : op.getOpOperands()) {
      int argIdx = getTensorIterArgIndex(operand.get(), forOp);
      if (argIdx >= 0) {
        usage.used.insert(argIdx);
      }
    }

    // Which iter_args does this op produce a new value for (yielded)?
    // yieldOp's i-th operand is the new value for the i-th iter_arg.
    for (OpResult opResult : op.getOpResults()) {
      for (OpOperand &yieldOperand : yieldOp->getOpOperands()) {
        if (yieldOperand.get() == opResult) {
          unsigned argIdx = yieldOperand.getOperandNumber();
          if (isa<RankedTensorType>(
                  forOp.getRegionIterArgs()[argIdx].getType())) {
            usage.updated.insert(argIdx);
          }
          break;
        }
      }
    }
  }

  return result;
}

// Returns true if `outNode` updates a tensor iter_arg that `inNode` uses.
// This is the "edge" in the bipartite fully-connected check: an out
// produces a new value for the iter_arg (yielded), and an in consumes
// the iter_arg on the next iteration. Both sides must touch the same
// iter_arg index for the edge to exist.
static bool hasUpdateUseConnection(
    BlockNodeInfo *outNode, BlockNodeInfo *inNode,
    const llvm::DenseMap<int, BlockIterArgUsage> &usageMap)
{
  auto outIt = usageMap.find(outNode->blockId);
  auto inIt = usageMap.find(inNode->blockId);
  if (outIt == usageMap.end() || inIt == usageMap.end()) {
    return false;
  }
  for (unsigned argIdx : outIt->second.updated) {
    if (inIt->second.used.contains(argIdx)) {
      return true;
    }
  }
  return false;
}

// Partition degree-zero nodes by mainloopId so the per-mainloop check
// in isScopeSerialExecution can look them up in O(1).
static void partitionDegreeZeroByMainloop(
    const DegreeInfo &degreeInfo,
    llvm::DenseMap<int, llvm::SmallVector<BlockNodeInfo *, 4>> &inByMainloop,
    llvm::DenseMap<int, llvm::SmallVector<BlockNodeInfo *, 4>> &outByMainloop)
{
  for (BlockNodeInfo *n : degreeInfo.inDegreeZero) {
    inByMainloop[n->mainloopId].push_back(n);
  }
  for (BlockNodeInfo *n : degreeInfo.outDegreeZero) {
    outByMainloop[n->mainloopId].push_back(n);
  }
}

// Per-mainloop fully-connected check. Returns true when every
// (out, in) pair is connected by an iter_arg-update edge, or when one
// side is empty (trivially serial).
static bool checkMainloopFullyConnected(
    llvm::StringRef scope, int mid,
    llvm::ArrayRef<BlockNodeInfo *> inZero,
    llvm::ArrayRef<BlockNodeInfo *> outZero,
    scf::ForOp forOp)
{
  if (inZero.empty() || outZero.empty()) {
    LDBG("[INFO]: " << scope << " mainloopId="
                    << mid << ": inDegZero=" << inZero.size()
                    << ", outDegZero=" << outZero.size()
                    << " -> trivially serial (one side empty)");
    return true;
  }

  auto usageMap = collectBlockIterArgUsage(forOp);

  for (BlockNodeInfo *outNode : outZero) {
    for (BlockNodeInfo *inNode : inZero) {
      // Self-edge (outNode == inNode) is trivially present: the same
      // block is both the entry and exit of the dataflow subgraph, so
      // the data flows through itself. No cross-block iter_arg check
      // is needed -- the strict hasUpdateUseConnection check below is
      // only meaningful for genuine cross-block (out != in) edges.
      if (outNode == inNode) {
        continue;
      }
      if (!hasUpdateUseConnection(outNode, inNode, usageMap)) {
        LDBG("[WARN]: " << scope << " mainloopId="
                        << mid << ": no iter_arg edge from outDegZero blockId="
                        << outNode->blockId << " to inDegZero blockId="
                        << inNode->blockId);
        LDBG("[INFO]: " << scope << " mainloopId="
                        << mid << ": NOT fully connected -> NOT serial");
        return false;
      }
    }
  }

  LDBG("[INFO]: " << scope << " mainloopId="
                  << mid << ": fully connected -> serial");
  return true;
}

// Returns true if the mainloop identified by `mainloopId` is serial:
// both its VECTOR and CUBE main_loops are fully connected through
// their iter_arg-update edges. A mainloopId without a matching VECTOR
// or CUBE forOp is treated as trivially serial on the missing side.
static bool isMainloopSerial(
    int mainloopId, llvm::ArrayRef<MainLoopData> mainLoops,
    const llvm::DenseMap<int, llvm::SmallVector<BlockNodeInfo *, 4>>
        &vectorInByMainloop,
    const llvm::DenseMap<int, llvm::SmallVector<BlockNodeInfo *, 4>>
        &vectorOutByMainloop,
    const llvm::DenseMap<int, llvm::SmallVector<BlockNodeInfo *, 4>>
        &cubeInByMainloop,
    const llvm::DenseMap<int, llvm::SmallVector<BlockNodeInfo *, 4>>
        &cubeOutByMainloop)
{
  const MainLoopData *vectorData = nullptr;
  const MainLoopData *cubeData = nullptr;
  for (const MainLoopData &data : mainLoops) {
    if (data.mainloopId != mainloopId) {
      continue;
    }
    if (data.scope == "VECTOR") {
      vectorData = &data;
    } else if (data.scope == "CUBE") {
      cubeData = &data;
    }
  }

  auto lookupZero = [](
      const llvm::DenseMap<int, llvm::SmallVector<BlockNodeInfo *, 4>>
          &byMainloop,
      int mid) {
    auto it = byMainloop.find(mid);
    if (it == byMainloop.end()) {
      return llvm::ArrayRef<BlockNodeInfo *>{};
    }
    return llvm::ArrayRef<BlockNodeInfo *>(it->second);
  };

  bool vectorSerial = true;
  if (vectorData) {
    llvm::ArrayRef<BlockNodeInfo *> inZero = lookupZero(vectorInByMainloop, mainloopId);
    llvm::ArrayRef<BlockNodeInfo *> outZero = lookupZero(vectorOutByMainloop, mainloopId);
    vectorSerial = checkMainloopFullyConnected(
        "VECTOR", mainloopId, inZero, outZero, vectorData->forOp);
  }

  bool cubeSerial = true;
  if (cubeData) {
    llvm::ArrayRef<BlockNodeInfo *> inZero = lookupZero(cubeInByMainloop, mainloopId);
    llvm::ArrayRef<BlockNodeInfo *> outZero = lookupZero(cubeOutByMainloop, mainloopId);
    cubeSerial = checkMainloopFullyConnected(
        "CUBE", mainloopId, inZero, outZero, cubeData->forOp);
  }

  return vectorSerial && cubeSerial;
}

// Build the dataflow graph (per-mainloop block nodes) and 
// the module-wide tcb_id lookup. The graph is just adjacency info
// without inter-core edges.
static void collectDataflowGraph(
    ModuleOp module,
    llvm::SmallVectorImpl<MainLoopData> &mainLoops,
    llvm::DenseMap<Value, int> &valueToTcbId)
{
  buildDataFlowGraph(module, mainLoops);
  buildMarkValueToTcbIdMap(module, valueToTcbId);
}

// Link a single (VECTOR, CUBE) mainloop pair's out-edges in both
// directions: VECTOR -> CUBE (via CopyOp) and CUBE -> VECTOR (via FixpipeOp).
static void linkOneMainloopPair(
    MainLoopData *vectorData, MainLoopData *cubeData,
    const llvm::DenseMap<Value, int> &valueToTcbId)
{
  // tcb_id -> CUBE BlockNode* (which CUBE block consumes each buffer).
  // CUBE consumes its alloc directly (no cast tracing).
  llvm::DenseMap<int, BlockNodeInfo *> tcbToCubeNode;
  buildTcbConsumerMap(cubeData->forOp, cubeData->blockNodes,
                      valueToTcbId, /*traceCasts=*/false, tcbToCubeNode);

  // tcb_id -> VECTOR BlockNode* (which VECTOR block consumes each buffer).
  // VECTOR reads the CUBE-produced alloc through memref/tensor casts.
  llvm::DenseMap<int, BlockNodeInfo *> tcbToVectorNode;
  buildTcbConsumerMap(vectorData->forOp, vectorData->blockNodes,
                      valueToTcbId, /*traceCasts=*/true, tcbToVectorNode);

  populateOutNodesViaTransferOp<hivm::CopyOp>(
      vectorData->forOp, vectorData->blockNodes, valueToTcbId, tcbToCubeNode);
  populateOutNodesViaTransferOp<hivm::FixpipeOp>(
      cubeData->forOp, cubeData->blockNodes, valueToTcbId, tcbToVectorNode);
}

// Pair each VECTOR mainloop with its matching CUBE mainloop
// by mainloopId, then populate the out-edges of both sides.
static void linkDataflowEdges(
    llvm::SmallVectorImpl<MainLoopData> &mainLoops,
    const llvm::DenseMap<Value, int> &valueToTcbId)
{
  // Group MainLoopData by mainloopId, separated by scope. VECTOR and CUBE
  // main_loops with the same mainloopId are a matched pair.
  llvm::DenseMap<int, MainLoopData *> vectorLoopById;
  llvm::DenseMap<int, MainLoopData *> cubeLoopById;
  for (MainLoopData &data : mainLoops) {
    if (data.scope == "VECTOR") {
      vectorLoopById[data.mainloopId] = &data;
    } else if (data.scope == "CUBE") {
      cubeLoopById[data.mainloopId] = &data;
    }
  }

  for (auto &entry : vectorLoopById) {
    int mainloopId = entry.first;
    MainLoopData *vectorData = entry.second;
    auto cubeIt = cubeLoopById.find(mainloopId);
    if (cubeIt == cubeLoopById.end()) {
      LDBG("[WARN]: No matching CUBE main_loop for mainloopId="
           << mainloopId << "; skipping out-edge population.");
      continue;
    }
    linkOneMainloopPair(vectorData, cubeIt->second, valueToTcbId);
  }
}

// Debug print of the full dataflow graph (one log line per node + per
// outgoing edge).
static void printDataflowGraph(llvm::ArrayRef<MainLoopData> mainLoops)
{
  size_t total = 0;
  for (const MainLoopData &data : mainLoops) {
    total += data.blockNodes.size();
  }
  LDBG("[INFO]: Collected " << total
                            << " block nodes with inter-core data transfer");
  for (const MainLoopData &data : mainLoops) {
    for (const BlockNodeInfo &node : data.blockNodes) {
      LDBG("  mainloopId=" << node.mainloopId
                           << ", scope=" << node.belongingScope
                           << ", blockId=" << node.blockId
                           << ", outNodes.size=" << node.outNodes.size());
      for (BlockNodeInfo *out : node.outNodes) {
        LDBG("    -> (mainloopId=" << out->mainloopId
                                   << ", scope=" << out->belongingScope
                                   << ", blockId=" << out->blockId << ")");
      }
    }
  }
}

// Compute in-degree-0 / out-degree-0 nodes for VECTOR and CUBE.
// A node is inDegreeZero when no other same-scope node can reach it
// (possibly through nodes of the other scope); symmetric for outDegreeZero.
// These are the entry/exit points of the scope-induced dataflow chain.
static void findScopeBoundaries(
    llvm::ArrayRef<MainLoopData> mainLoops,
    DegreeInfo &vectorDegree, DegreeInfo &cubeDegree)
{
  vectorDegree = findDegreeZeroNodesInScope(mainLoops, "VECTOR");
  cubeDegree = findDegreeZeroNodesInScope(mainLoops, "CUBE");
  logDegreeInfo("VECTOR", vectorDegree);
  logDegreeInfo("CUBE", cubeDegree);
}

// Serial-execution check. A mainloop is "serial" when, for
// both its VECTOR and CUBE forOps, every out-degree-0 node updates a
// tensor iter_arg that every in-degree-0 node uses (bipartite fully
// connected under the iter_arg-update edge). The overall result is the
// AND across every mainloopId: the CV pipeline is only considered
// unnecessary (i.e. we fall back) when ALL main_loop forOps are
// serial. A single non-serial mainloopId keeps the CV pipeline active.
static bool checkSerialExecution(
    llvm::ArrayRef<MainLoopData> mainLoops,
    const DegreeInfo &vectorDegree, const DegreeInfo &cubeDegree)
{
  // Per-scope, per-mainloopId degree-zero maps.
  llvm::DenseMap<int, llvm::SmallVector<BlockNodeInfo *, 4>> vectorInByMainloop;
  llvm::DenseMap<int, llvm::SmallVector<BlockNodeInfo *, 4>> vectorOutByMainloop;
  llvm::DenseMap<int, llvm::SmallVector<BlockNodeInfo *, 4>> cubeInByMainloop;
  llvm::DenseMap<int, llvm::SmallVector<BlockNodeInfo *, 4>> cubeOutByMainloop;
  partitionDegreeZeroByMainloop(vectorDegree, vectorInByMainloop,
                                vectorOutByMainloop);
  partitionDegreeZeroByMainloop(cubeDegree, cubeInByMainloop, cubeOutByMainloop);

  // All unique mainloopIds across both scopes.
  llvm::DenseSet<int> allMainloopIds;
  for (const MainLoopData &data : mainLoops) {
    allMainloopIds.insert(data.mainloopId);
  }

  bool allSerial = true;
  for (int mid : allMainloopIds) {
    bool isSerial = isMainloopSerial(mid, mainLoops, vectorInByMainloop,
                                     vectorOutByMainloop, cubeInByMainloop,
                                     cubeOutByMainloop);
    LDBG("[INFO]: mainloopId=" << mid << " serial: "
                               << (isSerial ? "YES" : "NO"));
    if (!isSerial) {
      allSerial = false;
    }
  }

  LDBG("[INFO]: All main_loops serial: " << (allSerial ? "YES" : "NO"));
  
  if(allSerial) {
    LDBG("[INFO]: All executions in mainloop are serial!");
  }

  return allSerial;
}

static bool checkMainLoopDataflow(ModuleOp module)
{
  llvm::SmallVector<MainLoopData, 4> mainLoops;
  llvm::DenseMap<Value, int> valueToTcbId;
  collectDataflowGraph(module, mainLoops, valueToTcbId);

  linkDataflowEdges(mainLoops, valueToTcbId);
  printDataflowGraph(mainLoops);

  DegreeInfo vectorDegree;
  DegreeInfo cubeDegree;
  findScopeBoundaries(mainLoops, vectorDegree, cubeDegree);

  return checkSerialExecution(mainLoops, vectorDegree, cubeDegree);
}

} // namespace

// Check "use-after-update" patterns on tensor iter_args in main_loop forOp.
bool checkTensorArgsUseAfterUpdate(ModuleOp module)
{
  bool hasViolation = false;
  CVPipeline::walkMainLoopForOps(module, [&](scf::ForOp forOp, int /*mainloopId*/) -> WalkResult {
    Block *body = forOp.getBody();
    if (!body) {
      return WalkResult::advance();
    }
    auto yieldOp = cast<scf::YieldOp>(body->getTerminator());

    for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
      if (!isa<RankedTensorType>(forOp.getRegionIterArgs()[i].getType())) {
        continue;
      }
      Operation *defOp = yieldOp.getOperand(i).getDefiningOp();
      if (!defOp) {
        continue;
      }
      Value iterArg = forOp.getRegionIterArgs()[i];

      for (Operation *user : iterArg.getUsers()) {
        if (user == yieldOp) {
          continue;
        }
        // The user may live in a nested region inside the forOp body
        // (e.g. inside an scf.if). Walk up to the top-level op in
        // the body so we can compare program order with defOp.
        Operation *topLevelUser = user;
        while (topLevelUser && topLevelUser->getParentOp() != forOp) {
          topLevelUser = topLevelUser->getParentOp();
        }
        if (!topLevelUser || topLevelUser->getBlock() != body) {
          continue;
        }
        if (!defOp->isBeforeInBlock(topLevelUser)) {
          continue;
        }
        LDBG("[ERROR]: Exists tensor iter_args use after update!");
        hasViolation = true;
      }
    }
    return WalkResult::advance();
  });
  return hasViolation;
}

bool checkTensorArgsInMainLoop(ModuleOp module)
{
  bool shouldReturn = false;
  bool isExistIterArgs = false;

  CVPipeline::walkMainLoopForOps(module, [&](scf::ForOp forOp, int /*mainloopId*/) -> WalkResult {
    if (failed(hasTensorArgInDifferentBlockIds(forOp))) {
      isExistIterArgs = true;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (isExistIterArgs) {
    // Returns true when the dataflow is serial on both VECTOR/CUBE side
    shouldReturn = checkMainLoopDataflow(module);
  }

  return shouldReturn;
}

void AnalyzeArgsPass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("Before AnalyzeArgs:\n" << module << "\n");

  // if (failed(isInterceptedModule(module))) {
  //   return;
  // }
  
  if(checkTensorArgsUseAfterUpdate(module)) {
    signalPassFailure();
    return;
  }

  if (checkTensorArgsInMainLoop(module)) {
    CVPipeline::setFallbackAttr(module);
    signalPassFailure();
    return;
  }

  LDBG("After AnalyzeArgs:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeArgsPass()
{
  return std::make_unique<AnalyzeArgsPass>();
}

} // namespace triton
} // namespace mlir