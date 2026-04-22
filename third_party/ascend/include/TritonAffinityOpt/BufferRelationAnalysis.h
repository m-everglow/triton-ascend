/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#ifndef TRITON_AFFINITY_OPT_BUFFER_RELATION_ANALYSIS_H
#define TRITON_AFFINITY_OPT_BUFFER_RELATION_ANALYSIS_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {

// ============================================================================
// 全局单例：InnerMultibuffer 写入，AddIfControls / 下游 Pass 读取
//
// 使用方式：
//   1. InnerMultibuffer::runOnOperation:
//        getGlobalBufferRelation().clear();
//        for (auto &[forOp, selBuf, memrefs] : ...) {
//          getGlobalBufferRelation().addBufferRelation(forOp, selBuf, memrefs);
//        }
//
//   2. AddIfControls::runOnOperation:
//        auto &rel = getGlobalBufferRelation();
//        auto *bufferMap = rel.getBufferRelations(forOp);
//        for (auto &[selBuf, memrefs] : *bufferMap) { ... }
//
//   或者直接从 IR 重新构造（推荐）：
//        BufferRelationAnalysis analysis;
//        analysis.populateFromForOp(mainLoopForOp);
//        auto *bufferMap = analysis.getBufferRelations(mainLoopForOp);
// ============================================================================
class BufferRelationAnalysis {
public:
  using ForOpBufferMap = DenseMap<scf::ForOp, DenseMap<Value, SmallVector<Value>>>;
  using CrossCoreMap = DenseMap<Value, SmallVector<Value>>;

  BufferRelationAnalysis() = default;
  explicit BufferRelationAnalysis(Operation *op) {}

  // === Intra-core buffer relations ===
  void addBufferRelation(scf::ForOp forOp, Value selectedBuffer,
                         const SmallVector<Value> &memrefs) {
    bufferMap[forOp][selectedBuffer] = memrefs;
  }

  const DenseMap<Value, SmallVector<Value>> *
  getBufferRelations(scf::ForOp forOp) const {
    auto it = bufferMap.find(forOp);
    return it == bufferMap.end() ? nullptr : &it->second;
  }

  const ForOpBufferMap &getAllBufferRelations() const { return bufferMap; }
  bool empty() const { return bufferMap.empty(); }
  size_t size() const { return bufferMap.size(); }
  void clear() { bufferMap.clear(); crossCoreRelations.clear(); }

  // === 从 IR 重新构造 buffer relations（供下游 Pass 调用）===
  // InnerMultibuffer 在 IR 中插入了 scf.if + to_tensor 的 ping-pong 模式：
  //   %selectedBuffer = scf.if %cond -> (tensor<X>) {
  //     then:   %t = bufferization.to_tensor %memref0 : memref<X>
  //     else:   %e = bufferization.to_tensor %memref1 : memref<X>
  //   }
  void populateFromForOp(scf::ForOp forOp) {
    forOp.walk([&](scf::IfOp ifOp) {
      Value result = ifOp.getResult(0);
      if (!result || !mlir::isa<RankedTensorType>(result.getType()))
        return;

      SmallVector<Value> memrefs;
      ifOp.walk([&](Operation *op) {
        if (mlir::isa<bufferization::ToTensorOp>(op))
          memrefs.push_back(op->getOperand(0));
      });

      if (memrefs.size() == 2) {
        bufferMap[forOp][result] = memrefs;
      }
    });
  }

  // === Cross-core relations ===
  void addCrossCoreRelation(Value key, const SmallVector<Value> &values) {
    crossCoreRelations[key] = values;
  }

  const CrossCoreMap &getCrossCoreRelations() const { return crossCoreRelations; }

  const SmallVector<Value> *getCrossCoreRelation(Value key) const {
    auto it = crossCoreRelations.find(key);
    return it == crossCoreRelations.end() ? nullptr : &it->second;
  }

  void clearCrossCoreRelations() { crossCoreRelations.clear(); }

private:
  ForOpBufferMap bufferMap;
  CrossCoreMap crossCoreRelations;
};

// ============================================================================
// 全局单例访问函数
// ============================================================================
inline BufferRelationAnalysis &getGlobalBufferRelation() {
  static BufferRelationAnalysis instance;
  return instance;
}

} // namespace triton
} // namespace mlir

#endif
