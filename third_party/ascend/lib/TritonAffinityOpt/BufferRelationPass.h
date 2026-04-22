// BufferRelationParser.h
// 使用示例

#ifndef BUFFER_RELATION_PARSER_H
#define BUFFER_RELATION_PARSER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include <string>

namespace mlir {
namespace triton {

// Buffer relation: selected_dep -> {new_buf0, new_buf1}
struct BufferRelation {
    Value selectedDep;   // 消费者 select 的结果
    Value newBuf0;       // 生产者 buf0 select 的结果
    Value newBuf1;       // 生产者 buf1 select 的结果
};

// ForOpSelectedBufferMap: 供上下游 Pass 共享的数据结构
//   Key: scf::ForOp — 插入 double buffer 的那个 for 循环
//   Inner Key: Value — selectedBuffer（scf.if 的结果，consumer 替换后使用的值）
//   Value: SmallVector<Value> — memref 列表（如 {memspacecast, memspacecast_1}）
using ForOpSelectedBufferMap =
    llvm::DenseMap<scf::ForOp, llvm::DenseMap<Value, llvm::SmallVector<Value>>>;

// forOp -> [buffer relations]
using BufferRelationMap = llvm::DenseMap<scf::ForOp, llvm::SmallVector<BufferRelation, 2>>;

/**
 * collectBufferRelations - 从 forOp 中解析 double buffer 的消费者-生产者关系
 *
 * 识别模式:
 *   select.condition -> (tensor.splat) -> (arith.cmpi) -> (arith.remsi)
 *
 * 使用方式:
 *   SmallVector<BufferRelation, 2> relations = collectBufferRelations(forOp);
 */
SmallVector<BufferRelation, 2> collectBufferRelations(scf::ForOp forOp) {
    SmallVector<BufferRelation, 2> relations;

    if (!forOp) return relations;

    // 遍历所有操作，收集符合 buffer select 链模式的 select ops
    for (auto &op : forOp.getBody()->getOperations()) {
        auto selectOp = dyn_cast<arith::SelectOp>(op);
        if (!selectOp) continue;

        // 检查 condition 是否来自 tensor.splat -> cmpi -> remsi
        auto condition = selectOp.getCondition();
        auto splatOp = condition.getDefiningOp<tensor::SplatOp>();
        if (!splatOp) continue;

        auto cmpiOp = splatOp.getOperand(0).getDefiningOp<arith::CmpIOp>();
        if (!cmpiOp) continue;

        auto remsiOp = cmpiOp.getOperand(0).getDefiningOp<arith::RemSIOp>();
        if (!remsiOp) continue;

        // 这是 buffer select 链
        auto trueValue = selectOp.getTrueValue();
        auto falseValue = selectOp.getFalseValue();
        auto trueDefOp = trueValue.getDefiningOp();
        auto falseDefOp = falseValue.getDefiningOp();

        // 消费者: 两个 operands 都是 select 结果 (new_buf0, new_buf1)
        if (isa<arith::SelectOp>(trueDefOp) && isa<arith::SelectOp>(falseDefOp)) {
            BufferRelation rel;
            rel.selectedDep = selectOp.getResult();
            rel.newBuf0 = trueValue;
            rel.newBuf1 = falseValue;
            relations.push_back(rel);
        }
    }

    return relations;
}

/**
 * parseBufferRelations - 从 Module 中解析所有 buffer 关系
 *
 * 使用方式:
 *   BufferRelationMap map = parseBufferRelations(module);
 *   for (auto &p : map) {
 *       scf::ForOp forOp = p.first;
 *       auto &relations = p.second;
 *       // ...
 *   }
 */
BufferRelationMap parseBufferRelations(ModuleOp module) {
    BufferRelationMap result;

    module.walk([&](scf::ForOp forOp) {
        if (!forOp->hasAttr("ssbuffer.mainloop"))
            return WalkResult::advance();

        auto relations = collectBufferRelations(forOp);
        if (!relations.empty())
            result[forOp] = relations;

        return WalkResult::advance();
    });

    return result;
}

} // namespace triton
} // namespace mlir

#endif // BUFFER_RELATION_PARSER_H

// ========== 使用示例 ==========
/*
// 在 MLIR Pass 中使用:

#include "BufferRelationParser.h"

namespace mlir {
namespace triton {

struct MyPass : public OperationPass<MyPass, ModuleOp> {
    void runOnOperation() override {
        ModuleOp module = getOperation();

        // 解析所有 buffer 关系
        auto relationMap = parseBufferRelations(module);

        // 遍历每个 forOp 及其 buffer 关系
        for (auto &p : relationMap) {
            scf::ForOp forOp = p.first;
            auto &relations = p.second;

            llvm::errs() << "ForOp: " << forOp << "\n";

            for (auto &rel : relations) {
                llvm::errs() << "  Consumer: " << rel.selectedDep << "\n";
                llvm::errs() << "    Producer buf0: " << rel.newBuf0 << "\n";
                llvm::errs() << "    Producer buf1: " << rel.newBuf1 << "\n";
            }
        }
    }
};

} // namespace triton
} // namespace mlir
*/