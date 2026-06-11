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

#ifndef TRITON_ASCEND_ANALYZE_ARGS_H
#define TRITON_ASCEND_ANALYZE_ARGS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <string>

namespace mlir {
namespace triton {

// Data structures shared across the AnalyzeArgs pass. They describe the
// per-block and per-main_loop state collected during the dataflow graph
// analysis. Kept in the header so other tools / TUs can consume them
// without re-declaring the same shape.

// Adjacency-list node describing one compute block that participates in
// inter-core (VECTOR<->CUBE) data transfer within a main_loop forOp.
struct BlockNodeInfo {
  int mainloopId = 0;
  std::string belongingScope;
  int blockId = 0;
  llvm::SmallVector<BlockNodeInfo *, 4> outNodes;
};

// Data collected for each tensor iter_arg: the first block_id that uses it.
// (We rely on checkTensorArgsUseAfterUpdate to guarantee every subsequent use
// happens before the iter_arg is updated, so the "first use" block_id is the
// only piece of block_id context we need.)
struct TensorArgBlockInfo {
  int firstBlockId = -1;
};

// Per-main_loop collected state. One entry per forOp marked with
// `ssbuffer.main_loop`, carrying the block nodes of that main_loop and
// enough context (the forOp itself + scope) to do further cross-scope
// linking later.
struct MainLoopData {
  scf::ForOp forOp;
  std::string scope;
  int mainloopId = 0;
  llvm::SmallVector<BlockNodeInfo, 8> blockNodes;
};

// The "boundary" nodes of one scope in the dataflow graph:
//   - inDegreeZero:  entry-point nodes (no other same-scope node can reach
//                    them via the dataflow graph)
//   - outDegreeZero: exit-point nodes (they cannot reach any other
//                    same-scope node)
// Used for both VECTOR and CUBE sides.
struct DegreeInfo {
  llvm::SmallVector<BlockNodeInfo *, 4> inDegreeZero;
  llvm::SmallVector<BlockNodeInfo *, 4> outDegreeZero;
};

// Per-block tensor iter_arg usage inside one main_loop:
//   updated: indices of tensor iter_args whose new value is produced by
//            this block (its op's result is yielded by the forOp)
//   used:    indices of tensor iter_args that this block consumes as
//            operands on the next iteration
struct BlockIterArgUsage {
  llvm::DenseSet<unsigned> updated;
  llvm::DenseSet<unsigned> used;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_ASCEND_ANALYZE_ARGS_H
