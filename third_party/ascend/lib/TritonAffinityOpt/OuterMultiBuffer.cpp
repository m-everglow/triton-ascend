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

// Pass: OuterMultibuffer (--outer-multibuffer)
//
// 功能：SSBUF 核间（CV间）多缓存分配（设计文档 §6.4.3）
//
// 核心思路：
//   昇腾 AI Core 内有 CUBE 核（矩阵运算）和 VECTOR 核（向量运算），两者通过
//   hivm.hir.sync_block_set / hivm.hir.sync_block_wait 做 flag 握手。
//   单 buffer 方案中，CUBE 写完数据 → set flag → VECTOR wait flag → 读数据，
//   是串行的。多缓存（double buffer）让 CUBE 写 buf0 时 VECTOR 同时读 buf1，
//   两者用不同的 flag ID 来区分：counter%N==0 用 flag_base，否则用 flag_base+1。
//
// 识别方法：
//   在 CUBE scope 和 VECTOR scope 的 scf.for body 内，找带相同 ssbuffer.id 属性
//   的 hivm.hir.sync_block_set / sync_block_wait op（由上游 DAGSync/DAGScope
//   pass 插入，并标注 ssbuffer.id 以标识同一 C→V 或 V→C 传输组）。
//
// 转换过程（对每个 ssbuffer.id 组）：
//   Step 1. findTransferGroups: 扫描 CUBE/VECTOR scope，按 ssbuffer.id 分组收集
//           hivm sync op 和 transfer op (fixpipe/hir.copy)，匹配同 id 的
//           CUBE 侧 ops 和 VECTOR 侧 ops 构成 pair。
//   Step 2. createBuffers: 在 scf.for 外部创建 N 个 buffer（memref.alloc）
//   Step 3. wrapTransferOps: 将 fixpipe/hir.copy 包装在 scf.if 中，根据
//           counter % N 选择对应的 buffer
//   Step 4. handleCounter: 将 counter 作为 iter_args 传递，在 loop 结束时更新

#define GEN_PASS_DEF_OUTERMULTIBUFFER
#include "TritonAffinityOpt/Passes.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/DenseMap.h"
#include "TritonAffinityOpt/BufferRelationAnalysis.h"
#include "TritonSyncScope/Common/FlagIdManager.h"

namespace mlir {
namespace triton {
#include "ascend/include/TritonAffinityOpt/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {

// =========================================================================
// 全局变量：记录 scf.if 的结果和对应的两个 buffer
// key: scf.if 的结果 Value (即 selected buffer)
// value: SmallVector<Value>{buffer0, buffer1}
// =========================================================================
static DenseMap<Value, SmallVector<Value>> crossCoreDependentMap;

// =========================================================================
// 数据结构
// =========================================================================

enum class Direction { C_TO_V, V_TO_C };

/// 一个 CV 间传输组：通过 ssbuffer.block_id + buffer 识别
struct TransferGroup {
    int ssbufferId;  // ssbuffer.block_id

    // C→V 传输对 (CUBE fixpipe -> VECTOR memory_space_cast)
    Operation *fixpipeOp;              // CUBE 侧: fixpipe (发端)
    Operation *memorySpaceCastOp;      // VECTOR 侧: memory_space_cast (收端)
    Value buffer0CtoV;                 // C→V 原始 buffer

    // V→C 传输对 (VECTOR hir.copy -> CUBE convert_layout)
    Operation *hirCopyOp;              // VECTOR 侧: hir.copy (发端)
    Operation *convertLayoutOp;        // CUBE 侧: convert_layout (收端)
    Value buffer0VtoC;                  // V→C 原始 buffer

    // 双 buffer (C→V 和 V→C 共用同一个 buffer1)
    Value buffer1;  // 新创建的双 buffer

    // 双 buffer 创建标志
    bool buffer1Created;

    // 原始的 sync op (用于后续删除)
    Operation *syncWaitOpCtoV;   // C→V 的 sync_block_wait (CUBE 侧)
    Operation *syncSetOpCtoV;   // C→V 的 sync_block_set (CUBE 侧)
    Operation *syncWaitOpVtoC;   // V→C 的 sync_block_wait (CUBE 侧)
    Operation *syncSetOpVtoC;   // V→C 的 sync_block_set (CUBE 侧)
};

// =========================================================================
// 辅助函数
// =========================================================================

/// 读取 ssbuffer.block_id 属性，无则返回 -1
static int getSsbufferId(Operation *op) {
    if (auto attr = op->getAttrOfType<IntegerAttr>("ssbuffer.block_id"))
        return static_cast<int>(attr.getInt());
    return -1;
}

// =========================================================================
// =========================================================================
// 辅助函数: reorderOpsBySsbufferIdTopo
// 将具有相同 ssbuffer.block_id 的操作聚集到一起
// 使用拓扑排序尊重 SSA 依赖关系
// block_id 小的组优先（即 block_id 5 的所有操作会在 block_id 7 的之前）
// 组内保持原始顺序
// =========================================================================
static void reorderOpsBySsbufferIdTopo(ModuleOp module) {
    module.walk([&](scope::ScopeOp scopeOp) {
        Block *body = scopeOp.getBody();
        if (!body || body->empty())
            return WalkResult::advance();

        // 跳过 terminator
        Operation *terminator = body->getTerminator();
        if (!terminator)
            return WalkResult::advance();

        // 收集所有带 ssbuffer.block_id 的 op 及其信息（不包括 terminator）
        SmallVector<std::tuple<int, int, Operation*>> opsWithBlockId; // (blockId, originalIdx, op)
        int idx = 0;
        for (auto &op : body->getOperations()) {
            if (&op == terminator)
                continue;
            int blockId = getSsbufferId(&op);
            if (blockId >= 0) {
                opsWithBlockId.push_back({blockId, idx, &op});
            }
            ++idx;
        }

        if (opsWithBlockId.empty())
            return WalkResult::advance();

        // 建立 block_id -> 操作列表 的映射
        DenseMap<int, SmallVector<Operation*>> blockIdToOps;
        for (auto &[blockId, origIdx, op] : opsWithBlockId) {
            blockIdToOps[blockId].push_back(op);
        }

        // 获取所有唯一的 block_id 并按值排序
        SmallVector<int> uniqueBlockIds;
        for (auto &p : blockIdToOps) {
            uniqueBlockIds.push_back(p.first);
        }
        std::sort(uniqueBlockIds.begin(), uniqueBlockIds.end());

        // 收集所有带 block_id 的操作指针
        SmallVector<Operation*> sortedOps;
        for (int blockId : uniqueBlockIds) {
            for (Operation *op : blockIdToOps[blockId]) {
                sortedOps.push_back(op);
            }
        }

        // 构建 SSA 依赖图：每个 sortedOp 依赖于哪些 sortedOp
        DenseMap<Operation*, SmallVector<Operation*>> internalDeps;
        for (Operation *op : sortedOps) {
            for (Value val : op->getOperands()) {
                if (auto *defOp = val.getDefiningOp()) {
                    if (llvm::is_contained(sortedOps, defOp)) {
                        internalDeps[op].push_back(defOp);
                    }
                }
            }
        }

        // 找到每个 sortedOp 的外部依赖位置（定义不在 sortedOps 中的操作）
        // 这些必须保持在 sortedOp 之前
        DenseMap<Operation*, int> externalDepPos;
        for (Operation *op : sortedOps) {
            int maxExtDepPos = -1;
            for (Value val : op->getOperands()) {
                if (auto *defOp = val.getDefiningOp()) {
                    if (!llvm::is_contained(sortedOps, defOp)) {
                        int depPos = std::distance(body->begin(), defOp->getIterator());
                        maxExtDepPos = std::max(maxExtDepPos, depPos);
                    }
                }
            }
            externalDepPos[op] = maxExtDepPos;
        }

        // 找到 sortedOps 中每个操作在其 block 中的原始位置
        DenseMap<Operation*, int> opToOrigIdx;
        idx = 0;
        for (auto &op : body->getOperations()) {
            if (&op == terminator)
                continue;
            opToOrigIdx[&op] = idx++;
        }

        // 找到第一个 sorted op 的原始位置作为基准
        int baseIdx = opToOrigIdx[sortedOps.front()];

        // 将所有 sortedOps 移动到 terminator 之前（按逆序以保持相对顺序）
        for (int i = static_cast<int>(sortedOps.size()) - 1; i >= 0; --i) {
            sortedOps[i]->moveBefore(terminator);
        }

        // 从 baseIdx 开始，逐个放置 sortedOps
        Block::iterator insertPoint = body->begin();
        std::advance(insertPoint, baseIdx);

        for (Operation *op : sortedOps) {
            // 计算目标位置：必须满足所有依赖
            int minPos = -1;

            // 内部依赖：必须在之后
            for (Operation *dep : internalDeps[op]) {
                int depPos = std::distance(body->begin(), dep->getIterator());
                minPos = std::max(minPos, depPos + 1);
            }

            // 外部依赖：必须在之后
            if (externalDepPos[op] >= 0) {
                minPos = std::max(minPos, externalDepPos[op] + 1);
            }

            Block::iterator targetPos = insertPoint;
            if (minPos >= baseIdx) {
                std::advance(targetPos, minPos - baseIdx);
            }

            op->moveBefore(body, targetPos);
        }

        llvm::errs() << "=== reorderOpsBySsbufferIdTopo: reordered "
                     << sortedOps.size() << " ops, " << uniqueBlockIds.size()
                     << " unique block_ids\n";
        return WalkResult::advance();
    });
}

// =========================================================================
// 辅助函数: findOriginalSyncOpInScope
// 在 transferOp 附近查找紧邻的 sync_block_wait/set op
// 优先查找紧邻 transferOp 的 sync op，因为它们是真正配套的
// 返回找到的 sync op，如果找到同时返回其 flag_id
// =========================================================================
static Operation* findOriginalSyncOpInScope(
    Operation *transferOp,
    bool isWait,  // true: 找 sync_block_wait, false: 找 sync_block_set
    bool isCtoV,  // true: C→V 传输, false: V→C 传输
    int64_t &outFlagId) {

    // 获取 transferOp 的 ssbuffer.block_id
    int ssbufferId = getSsbufferId(transferOp);
    if (ssbufferId < 0)
        return nullptr;

    // 确定要查找的 sync op 类型
    StringRef targetSyncName = isWait ? "sync_block_wait" : "sync_block_set";

    Block *block = transferOp->getBlock();
    if (!block)
        return nullptr;

    Operation *foundSyncOp = nullptr;
    outFlagId = -1;

    // 查找 transferOp 前后的 sync ops (在一定距离内的)
    // 向前查找最多 5 个操作
    auto it = transferOp->getIterator();
    for (int i = 0; i < 5 && it != block->begin(); ++i) {
        --it;
        Operation *op = &*it;
        StringRef opName = op->getName().getStringRef();
        if (!opName.contains(targetSyncName))
            continue;
        if (getSsbufferId(op) != ssbufferId)
            continue;
        // 获取 flag_id (优先 static_flag_id，回退到 flag)
        int64_t flagId = -1;
        if (auto flagAttr = op->getAttrOfType<IntegerAttr>("static_flag_id")) {
            flagId = static_cast<int64_t>(flagAttr.getInt());
        } else if (auto flagAttr = op->getAttrOfType<IntegerAttr>("flag")) {
            flagId = static_cast<int64_t>(flagAttr.getInt());
        }
        if (flagId >= 0) {
            outFlagId = flagId;
            foundSyncOp = op;
            break;  // 找到第一个就返回
        }
    }

    return foundSyncOp;
}

// =========================================================================
// 辅助函数: eraseOriginalSyncOpsInScope
// 删除与 transferOp 直接相邻的原始 sync_block_wait/set op
// 只删除紧挨着 transferOp 的 sync ops，而不是删除所有匹配的
// =========================================================================
static void eraseOriginalSyncOpsInScope(Operation *transferOp) {
    int ssbufferId = getSsbufferId(transferOp);
    if (ssbufferId < 0)
        return;

    Block *block = transferOp->getBlock();
    if (!block)
        return;

    SmallVector<Operation*> toErase;

    // 查找 transferOp 前后的 sync ops (在一定距离内的)
    // 向前查找最多 5 个操作
    auto it = transferOp->getIterator();
    for (int i = 0; i < 5 && it != block->begin(); ++i) {
        --it;
        Operation *op = &*it;
        StringRef opName = op->getName().getStringRef();
        bool isSyncOp = opName.contains("sync_block_wait") || opName.contains("sync_block_set");
        if (!isSyncOp)
            continue;
        if (getSsbufferId(op) != ssbufferId)
            continue;
        toErase.push_back(op);
    }

    // 向后查找最多 5 个操作
    it = std::next(transferOp->getIterator());
    for (int i = 0; i < 5 && it != block->end(); ++i, ++it) {
        Operation *op = &*it;
        StringRef opName = op->getName().getStringRef();
        bool isSyncOp = opName.contains("sync_block_wait") || opName.contains("sync_block_set");
        if (!isSyncOp)
            continue;
        if (getSsbufferId(op) != ssbufferId)
            continue;
        toErase.push_back(op);
    }

    // 删除找到的原始 sync ops
    for (auto *op : toErase) {
        op->erase();
    }
}

/// 判断是否是 hivm sync_block 系列 op
static bool isSyncBlockOp(Operation *op) {
    StringRef name = op->getName().getStringRef();
    return name.contains("sync_block_set") || name.contains("sync_block_wait");
}

/// 判断是否是 fixpipe 操作
static bool isFixpipeOp(Operation *op) {
    return op->getName().getStringRef().contains("fixpipe");
}

/// 判断是否是 hir.copy 操作
static bool isCopyOp(Operation *op) {
    return op->getName().getStringRef().contains("hir.copy");
}

/// 判断是否是 CV 间传输操作 (fixpipe 或 hir.copy)
static bool isTransferOp(Operation *op) {
    return isFixpipeOp(op) || isCopyOp(op);
}

/// 判断是否是 memref.memory_space_cast 操作 (VECTOR 侧接收 C→V 数据)
static bool isMemorySpaceCastOp(Operation *op) {
    return op->getName().getStringRef().contains("memory_space_cast");
}

/// 判断是否是 hivm.hir.convert_layout 操作 (CUBE 侧接收 V→C 数据)
static bool isConvertLayoutOp(Operation *op) {
    return op->getName().getStringRef().contains("convert_layout");
}

/// 判断是否是 CV 间接收操作 (memory_space_cast 或 convert_layout)
static bool isReceiveOp(Operation *op) {
    return isMemorySpaceCastOp(op) || isConvertLayoutOp(op);
}

/// 找到包含 opInScope 的最近 scope.scope 的 body Block
/// 在 ReturnOp 之前的位置插入（在所有操作之前）
static Block *getScopeInsertBlock(Operation *opInScope) {
    // 找到 scope.scope
    Operation *cur = opInScope->getParentOp();
    while (cur) {
        if (auto scopeOp = dyn_cast<scope::ScopeOp>(cur)) {
            Block *body = scopeOp.getBody();
            // 找到 ReturnOp 之前的位置（真正的开头）
            for (auto &op : body->getOperations()) {
                if (isa<scope::ReturnOp>(op)) {
                    // 返回 ReturnOp 之前的位置
                    return body;
                }
            }
            // 如果没有 ReturnOp，返回 body
            return body;
        }
        cur = cur->getParentOp();
    }
    return nullptr;
}

/// 找到 op 所在的 scope.body，然后找到第一个 scf.for 操作之前的位置
/// 用于在 scope 的开头（但在任何 for 循环之前）插入 buffer
static std::pair<Block *, Block::iterator> getScopeInsertPointBeforeFor(Operation *opInScope) {
    Operation *cur = opInScope->getParentOp();
    while (cur) {
        if (auto scopeOp = dyn_cast<scope::ScopeOp>(cur)) {
            Block *body = scopeOp.getBody();
            // 找到第一个 scf.for 之前的位置
            for (auto &op : body->getOperations()) {
                if (isa<scf::ForOp>(op)) {
                    // 返回该 forOp 之前的位置
                    return {body, op.getIterator()};
                }
            }
            // 如果没有 forOp，找 ReturnOp 之前
            for (auto &op : body->getOperations()) {
                if (isa<scope::ReturnOp>(op)) {
                    return {body, op.getIterator()};
                }
            }
            // 如果都没有，返回开头
            return {body, body->begin()};
        }
        cur = cur->getParentOp();
    }
    return {nullptr, Block::iterator()};
}

/// 找到包含 CUBE 或 VECTOR scope 的外层 scope（即 tt.func）
/// 两个 scope.scope (CUBE 和 VECTOR) 是并列的，都在 tt.func 内
/// 返回: {tt.func 的 body block, 在两个 scope.scope 之前的位置}
static std::pair<Block *, Block::iterator>
getOuterScopeInsertPointBeforeInnerScopes(Operation *opInInnerScope) {
    // Walk up the parent chain to find func.func
    Operation *cur = opInInnerScope;
    while (cur) {
        auto name = cur->getName().getStringRef();
        if (name.contains("func.func")) {
            break;
        }
        cur = cur->getParentOp();
    }

        if (!cur)
        return {nullptr, Block::iterator()};

    // Now we need to get the body of the tt.func operation
    // tt.func has a region with a body block
    Block *body = nullptr;
    if (cur->getNumRegions() > 0 && !cur->getRegion(0).empty()) {
        body = &cur->getRegion(0).front();
    }

    if (!body)
        return {nullptr, Block::iterator()};

    // 找到第一个 scope::ScopeOp 之前的位置（在两个并列 scope.scope 之前）
    for (auto &op : body->getOperations()) {
        if (isa<scope::ScopeOp>(op))
            return {body, op.getIterator()};
    }
    // 如果没有 scope::ScopeOp，找 ReturnOp 或开头
    for (auto &op : body->getOperations()) {
        if (isa<mlir::func::ReturnOp>(op))
            return {body, op.getIterator()};
    }
    return {body, body->begin()};
}

/// 找到 tt.func 级别在两个 scope.scope 之前的位置
/// 用于创建在两个内层 scope 之间共享的 buffer
static std::pair<Block *, Block::iterator>
getOuterScopeInsertPointBeforeOuterFor(Operation *opInInnerScope) {
    // 使用 getParentOfType 直接找到 tt.func
    auto funcOp = opInInnerScope->getParentOfType<mlir::func::FuncOp>();
    if (!funcOp)
        return {nullptr, Block::iterator()};

    Block *body = &funcOp.getBody().front();

    // 找到第一个 scope::ScopeOp 之前的位置
    for (auto &op : body->getOperations()) {
        if (isa<scope::ScopeOp>(op)) {
            return {body, op.getIterator()};
        }
    }
    // 如果没有 scope::ScopeOp，找 ReturnOp 或开头
    for (auto &op : body->getOperations()) {
        if (isa<mlir::func::ReturnOp>(op)) {
            return {body, op.getIterator()};
        }
    }
    return {body, body->begin()};
}

// =========================================================================
// Step 2e: wrapReceiveOpWithWaitIf
// 在接收操作之前插入 sync_block_wait
// wait 的 pipe 方向与发端相反（发端 setPipe=PIPE_FIX, waitPipe=PIPE_V）
// 收端需要 waitPipe=PIPE_FIX, setPipe=PIPE_V
// 如果提供了 precomputedCond，则使用它而不重新计算条件
// =========================================================================
static void wrapReceiveOpWithWaitIf(
    OpBuilder &builder, Location loc,
    Operation *receiveOp,
    Value counter, Value c2,
    int64_t flagId0, int64_t flagId1,
    bool isCtoV,
    Value precomputedCond = Value()) {
    // isCtoV: true 表示 C→V 接收 (memory_space_cast), false 表示 V→C 接收 (convert_layout)

    // 获取 ssbuffer.block_id
    int ssbufferId = getSsbufferId(receiveOp);

    MLIRContext *ctx = builder.getContext();

    // 收端的 pipe 方向与发端相反
    // C→V: CUBE set → VECTOR wait, 所以 VECTOR wait 时 setPipe=PIPE_V, waitPipe=PIPE_FIX
    // V→C: VECTOR set → CUBE wait, 所以 CUBE wait 时 setPipe=PIPE_V, waitPipe=PIPE_FIX
    auto setPipe = PipeAttr::get(ctx, hivm::PIPE::PIPE_V);
    auto waitPipe = PipeAttr::get(ctx, hivm::PIPE::PIPE_FIX);

    Value cond;
    if (precomputedCond) {
        // 使用预计算的条件
        cond = precomputedCond;
    } else {
        // 计算条件
        builder.setInsertionPoint(receiveOp);

        Type i32Type = counter.getType();
        Value c0 = builder.create<arith::ConstantOp>(
            loc, i32Type, builder.getI32IntegerAttr(0)).getResult();
        if (ssbufferId >= 0)
            c0.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
        Value idx = builder.create<arith::RemSIOp>(loc, counter, c2);
        if (ssbufferId >= 0)
            idx.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
        cond = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, idx, c0);
        if (ssbufferId >= 0)
            cond.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
    }

    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, /*withElse=*/true);
    if (ssbufferId >= 0)
        ifOp->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));

    // --- then branch: 使用 flagId0 ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block &thenBlock = ifOp.getThenRegion().front();
        builder.setInsertionPoint(&thenBlock.back());

        // 收端等待的是对方的 set，所以 core type 是对方
        auto waitCoreType = isCtoV ? hivm::TCoreType::CUBE : hivm::TCoreType::VECTOR;
        auto waitCoreAttr = hivm::TCoreTypeAttr::get(ctx, waitCoreType);
        auto waitOp0 = builder.create<hivm::SyncBlockWaitOp>(
            loc, waitCoreAttr, setPipe, waitPipe,
            builder.getI64IntegerAttr(flagId0));
        if (ssbufferId >= 0)
            waitOp0->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
    }

    // --- else branch: 使用 flagId1 ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block &elseBlock = ifOp.getElseRegion().front();
        builder.setInsertionPoint(&elseBlock.back());

        auto waitCoreType = isCtoV ? hivm::TCoreType::CUBE : hivm::TCoreType::VECTOR;
        auto waitCoreAttr = hivm::TCoreTypeAttr::get(ctx, waitCoreType);
        auto waitOp1 = builder.create<hivm::SyncBlockWaitOp>(
            loc, waitCoreAttr, setPipe, waitPipe,
            builder.getI64IntegerAttr(flagId1));
        if (ssbufferId >= 0)
            waitOp1->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
    }
}

// =========================================================================
// Step 2f: wrapReceiveOpWithReceiveIf
// 包装接收操作 (memory_space_cast / convert_layout)
// 这些操作有结果，需要正确处理 result replacement
// 返回创建的 scf::IfOp，以便后续操作可以在其结果后插入
// 注意：接收操作的 operand 不是 buffer，不需要替换
// 如果提供了 precomputedCond，则使用它而不重新计算条件
// =========================================================================
static scf::IfOp wrapReceiveOpWithReceiveIf(
    OpBuilder &builder, Location loc,
    Operation *receiveOp,
    Value counter, Value c2,
    ArrayRef<Value> buffers,
    Value precomputedCond = Value()) {

    int N = buffers.size();
    if (N == 0)
        return nullptr;

    // 获取 ssbuffer.block_id
    int ssbufferId = getSsbufferId(receiveOp);

    Value cond;
    if (precomputedCond) {
        // 使用预计算的条件
        cond = precomputedCond;
    } else {
        // 计算条件
        builder.setInsertionPoint(receiveOp);

        Type i32Type = counter.getType();
        Value c0 = builder.create<arith::ConstantOp>(
            loc, i32Type, builder.getI32IntegerAttr(0)).getResult();
        if (ssbufferId >= 0)
            c0.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
        Value idx = builder.create<arith::RemSIOp>(loc, counter, c2);
        if (ssbufferId >= 0)
            idx.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
        cond = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, idx, c0);
        if (ssbufferId >= 0)
            cond.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
    }

    // receiveOp 的结果类型
    SmallVector<Type> resultTypes;
    for (auto result : receiveOp->getResults())
        resultTypes.push_back(result.getType());

    // 创建 scf.if (不使用 auto-yield，手动管理)
    auto ifOp = builder.create<scf::IfOp>(loc, resultTypes, cond, /*withElse=*/true);
    if (ssbufferId >= 0)
        ifOp->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));

    // --- then branch: 使用 buffers[0] ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block *thenBlock = &ifOp.getThenRegion().front();
        builder.setInsertionPointToStart(thenBlock);

        Operation *clone0 = receiveOp->clone();
        // memory_space_cast 的 operand 0 是 buffer，需要替换
        if (clone0->getNumOperands() > 0)
            clone0->setOperand(0, buffers[0]);
        builder.insert(clone0);

        SmallVector<Value> yieldValues;
        for (auto result : clone0->getResults())
            yieldValues.push_back(result);
        builder.create<scf::YieldOp>(loc, yieldValues);
    }

    // --- else branch: 使用 buffers[1] ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block *elseBlock = &ifOp.getElseRegion().front();
        builder.setInsertionPointToStart(elseBlock);

        Operation *clone1 = receiveOp->clone();
        // memory_space_cast 的 operand 0 是 buffer，需要替换
        if (clone1->getNumOperands() > 0)
            clone1->setOperand(0, buffers[1]);
        builder.insert(clone1);

        SmallVector<Value> yieldValues;
        for (auto result : clone1->getResults())
            yieldValues.push_back(result);
        builder.create<scf::YieldOp>(loc, yieldValues);
    }

// 用 ifOp 的结果替换原 receive op 的结果
    for (size_t i = 0; i < receiveOp->getNumResults(); ++i) {
        receiveOp->getResult(i).replaceAllUsesWith(ifOp.getResult(i));
    }

    // 记录 crossCoreDependentMap: scf.if 的结果 -> {buffer0, buffer1}
    if (ifOp.getNumResults() > 0) {
        SmallVector<Value> bufferPair = {buffers[0], buffers[1]};
        crossCoreDependentMap[ifOp.getResult(0)] = bufferPair;
    }

    // 删除原始的 sync ops（在删除 receiveOp 之前）
    eraseOriginalSyncOpsInScope(receiveOp);

    // 删除原 op
    receiveOp->erase();

    return ifOp;
}

// =========================================================================
// Step 2g: wrapReceiveOpWithSetIf
// 在接收操作之后插入 sync_block_set
// set 的 pipe 方向与发端相反
// receiveOpResult 是 wrapReceiveOpWithReceiveIf 返回的 scf::IfOp 的结果值
// ssbufferId 是接收操作的 ssbuffer.block_id
// 如果提供了 precomputedCond，则使用它而不重新计算条件
// =========================================================================
static void wrapReceiveOpWithSetIf(
    OpBuilder &builder, Location loc,
    Value receiveOpResult,
    Value counter, Value c2,
    int64_t flagId0, int64_t flagId1,
    bool isCtoV,
    int ssbufferId,
    Value precomputedCond = Value()) {
    // isCtoV: true 表示 C→V 接收, false 表示 V→C 接收

    MLIRContext *ctx = builder.getContext();
    // 收端的 pipe 方向与发端相反
    auto setPipe = PipeAttr::get(ctx, hivm::PIPE::PIPE_V);
    auto waitPipe = PipeAttr::get(ctx, hivm::PIPE::PIPE_FIX);

    Value cond;
    if (precomputedCond) {
        // 使用预计算的条件
        cond = precomputedCond;
    } else {
        // 在 receiveOpResult 之后插入
        builder.setInsertionPointAfter(receiveOpResult.getDefiningOp());

        Type i32Type = counter.getType();
        Value c0 = builder.create<arith::ConstantOp>(
            loc, i32Type, builder.getI32IntegerAttr(0)).getResult();
        if (ssbufferId >= 0)
            c0.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
        Value idx = builder.create<arith::RemSIOp>(loc, counter, c2);
        if (ssbufferId >= 0)
            idx.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
        cond = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, idx, c0);
        if (ssbufferId >= 0)
            cond.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
    }

    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, /*withElse=*/true);
    if (ssbufferId >= 0)
        ifOp->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));

    // --- then branch: 使用 flagId0 ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block &thenBlock = ifOp.getThenRegion().front();
        builder.setInsertionPoint(&thenBlock.back());

        // 收端 set 是通知对方我读完了，所以 core type 是自己
        auto setCoreType = isCtoV ? hivm::TCoreType::VECTOR : hivm::TCoreType::CUBE;
        auto setCoreAttr = hivm::TCoreTypeAttr::get(ctx, setCoreType);
        auto setOp0 = builder.create<hivm::SyncBlockSetOp>(
            loc, setCoreAttr, setPipe, waitPipe,
            builder.getI64IntegerAttr(flagId0));
        if (ssbufferId >= 0)
            setOp0->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
    }

    // --- else branch: 使用 flagId1 ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block &elseBlock = ifOp.getElseRegion().front();
        builder.setInsertionPoint(&elseBlock.back());

        auto setCoreType = isCtoV ? hivm::TCoreType::VECTOR : hivm::TCoreType::CUBE;
        auto setCoreAttr = hivm::TCoreTypeAttr::get(ctx, setCoreType);
        auto setOp1 = builder.create<hivm::SyncBlockSetOp>(
            loc, setCoreAttr, setPipe, waitPipe,
            builder.getI64IntegerAttr(flagId1));
        if (ssbufferId >= 0)
            setOp1->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
    }
}

// =========================================================================
// Step 2e (original): wrapReceiveOpWithCounterIf
// 用于 memory_space_cast 和 convert_layout 等接收操作
// 这些操作有结果，需要正确处理 result replacement
// =========================================================================
static void wrapReceiveOpWithCounterIf(
    OpBuilder &builder, Location loc,
    Operation *receiveOp,
    Value counter, Value c2,
    ArrayRef<Value> buffers) {

    int N = buffers.size();
    if (N == 0)
        return;

    builder.setInsertionPoint(receiveOp);

    Type i32Type = counter.getType();
    Value c0 = builder.create<arith::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(0)).getResult();
    Value idx = builder.create<arith::RemSIOp>(loc, counter, c2);
    Value cond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, idx, c0);

    // receiveOp 的结果类型
    SmallVector<Type> resultTypes;
    for (auto result : receiveOp->getResults())
        resultTypes.push_back(result.getType());

    // 创建 scf.if (不使用 auto-yield，手动管理)
    auto ifOp = builder.create<scf::IfOp>(loc, resultTypes, cond, /*withElse=*/true);

    // --- then branch: 使用 buffers[0] ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block *thenBlock = &ifOp.getThenRegion().front();
        builder.setInsertionPointToStart(thenBlock);

        Operation *clone0 = receiveOp->clone();
        // receiveOp 的 operand 0 是输入 memref，用 buffer 替换
        if (clone0->getNumOperands() > 0) {
            clone0->setOperand(0, buffers[0]);
        }
        builder.insert(clone0);

        SmallVector<Value> yieldValues;
        for (auto result : clone0->getResults())
            yieldValues.push_back(result);
        builder.create<scf::YieldOp>(loc, yieldValues);
    }

    // --- else branch: 使用 buffers[1] ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block *elseBlock = &ifOp.getElseRegion().front();
        builder.setInsertionPointToStart(elseBlock);

        Operation *clone1 = receiveOp->clone();
        if (clone1->getNumOperands() > 0) {
            clone1->setOperand(0, buffers[1]);
        }
        builder.insert(clone1);

        SmallVector<Value> yieldValues;
        for (auto result : clone1->getResults())
            yieldValues.push_back(result);
        builder.create<scf::YieldOp>(loc, yieldValues);
    }

    // 用 ifOp 的结果替换原 receive op 的结果
    for (size_t i = 0; i < receiveOp->getNumResults(); ++i) {
        receiveOp->getResult(i).replaceAllUsesWith(ifOp.getResult(i));
    }

    // 删除原 op
    receiveOp->erase();
}

/// 找到最近的外层 scf::ForOp（如果不存在返回 nullptr）
static scf::ForOp findEnclosingForOp(Operation *op) {
    Operation *cur = op->getParentOp();
    while (cur) {
        if (auto forOp = dyn_cast<scf::ForOp>(cur))
            return forOp;
        cur = cur->getParentOp();
    }
    return nullptr;
}

/// 扫描 module 内所有 static_flag_id，返回下一个可用值（确保不冲突）
static int64_t getNextFlagId(ModuleOp module) {
    int64_t maxFlag = 1; // 预留 0, 1 给单 buffer 场景
    module.walk([&](Operation *op) {
        if (auto attr = op->getAttrOfType<IntegerAttr>("static_flag_id"))
            maxFlag = std::max(maxFlag, attr.getInt());
    });
    return maxFlag + 1;
}

// =========================================================================
// Step 1: findTransferGroups — 通过 buffer SSA value 匹配 CV 间传输对
//
// 通过 buffer 匹配传输组：
//   fixpipe outs(%buffer) <-> memory_space_cast %buffer
//   使用相同 buffer SSA value 的 fixpipe 和 memory_space_cast 是一对
//
// 同一传输组使用同一个 buffer
// =========================================================================
static SmallVector<TransferGroup>
findTransferGroups(ModuleOp module) {
    // 存储每个 buffer 的 defining op 对应的 op
    // C→V: fixpipe (CUBE) 和 memory_space_cast (VECTOR)
    // V→C: hir.copy (VECTOR) 和 convert_layout (CUBE)
    DenseMap<Operation*, Operation*> fixpipeByDefOp;
    DenseMap<Operation*, Operation*> memorySpaceCastByDefOp;
    DenseMap<Operation*, Operation*> hirCopyByDefOp;
    DenseMap<Operation*, Operation*> convertLayoutByDefOp;
    DenseMap<Operation*, Value> bufferValueByDefOp;  // 记录 defining op 对应的 buffer value
    DenseMap<Operation*, int> ssbufferIdByDefOp;  // 记录 defining op 对应的 ssbuffer.block_id

    // 存储 sync ops (通过 ssbufferId 索引)
    // CUBE 侧: sync_wait C→V, sync_set C→V, sync_wait V→C, sync_set V→C
    // VECTOR 侧: sync_wait C→V, sync_set C→V, sync_wait V→C, sync_set V→C
    DenseMap<int, Operation*> syncWaitCtoVCube;
    DenseMap<int, Operation*> syncSetCtoVCube;
    DenseMap<int, Operation*> syncWaitVtoCCube;
    DenseMap<int, Operation*> syncSetVtoCCube;
    DenseMap<int, Operation*> syncWaitCtoVVector;
    DenseMap<int, Operation*> syncSetCtoVVector;
    DenseMap<int, Operation*> syncWaitVtoCVector;
    DenseMap<int, Operation*> syncSetVtoCVector;

    // Walk through FuncOp first, then find scope::ScopeOp inside
    module.walk([&](Operation *op) {
        if (!op->getName().getStringRef().contains("func.func"))
            return WalkResult::advance();

        op->walk([&](scope::ScopeOp scope) {
            auto tcoreAttr = scope->getAttrOfType<TCoreTypeAttr>("hivm.tcore_type");
            if (!tcoreAttr)
                return WalkResult::advance();

            bool isCube   = (tcoreAttr.getTcoretype() == TCoreType::CUBE);
            bool isVector = (tcoreAttr.getTcoretype() == TCoreType::VECTOR);
            if (!isCube && !isVector)
                return WalkResult::advance();

            scope.walk([&](Operation *opInner) {
                int id = getSsbufferId(opInner);
                if (id < 0)
                    return;

                // CUBE scope: fixpipe (发端) 和 convert_layout (收端)
                if (isCube) {
                    // fixpipe: operand 1 是 output buffer
                    if (isFixpipeOp(opInner) && opInner->getNumOperands() > 1) {
                        Value buffer = opInner->getOperand(1);
                        Operation *defOp = buffer.getDefiningOp();
                        if (defOp) {
                            fixpipeByDefOp[defOp] = opInner;
                            bufferValueByDefOp[defOp] = buffer;
                            ssbufferIdByDefOp[defOp] = id;
                        }
                    }
                    // convert_layout: operand 0 是 buffer (V→C 收端)
                    if (isConvertLayoutOp(opInner) && opInner->getNumOperands() > 0) {
                        Value buffer = opInner->getOperand(0);
                        Operation *defOp = buffer.getDefiningOp();
                        if (defOp) {
                            convertLayoutByDefOp[defOp] = opInner;
                            bufferValueByDefOp[defOp] = buffer;
                            ssbufferIdByDefOp[defOp] = id;
                        }
                    }
                }

                // VECTOR scope: hir.copy (发端) 和 memory_space_cast (收端)
                if (isVector) {
                    // hir.copy: operand 1 是 output buffer
                    if (isCopyOp(opInner) && opInner->getNumOperands() > 1) {
                        Value buffer = opInner->getOperand(1);
                        Operation *defOp = buffer.getDefiningOp();
                        if (defOp) {
                            hirCopyByDefOp[defOp] = opInner;
                            bufferValueByDefOp[defOp] = buffer;
                            ssbufferIdByDefOp[defOp] = id;
                        }
                    }
                    // memory_space_cast: operand 0 是 buffer (C→V 收端)
                    if (isMemorySpaceCastOp(opInner) && opInner->getNumOperands() > 0) {
                        Value buffer = opInner->getOperand(0);
                        Operation *defOp = buffer.getDefiningOp();
                        if (defOp) {
                            memorySpaceCastByDefOp[defOp] = opInner;
                            bufferValueByDefOp[defOp] = buffer;
                            ssbufferIdByDefOp[defOp] = id;
                        }
                    }
                }
            });

            return WalkResult::advance();
        });

        return WalkResult::advance();
    });

    SmallVector<TransferGroup> groups;

    // =============================================================
    // C→V 传输对: fixpipe (CUBE) <-> memory_space_cast (VECTOR)
    // =============================================================

    // 处理有 fixpipe 的 C→V 组
    for (auto &p : fixpipeByDefOp) {
        Operation *defOp = p.first;
        TransferGroup g;
        g.ssbufferId = ssbufferIdByDefOp[defOp];
        g.fixpipeOp = p.second;
        g.buffer0CtoV = bufferValueByDefOp[defOp];
        g.memorySpaceCastOp = memorySpaceCastByDefOp.count(defOp) ?
                              memorySpaceCastByDefOp[defOp] : nullptr;
        g.hirCopyOp = nullptr;
        g.convertLayoutOp = nullptr;
        g.buffer0VtoC = Value();  // C→V 组没有 V→C buffer，初始化为空
        g.buffer1Created = false;

        groups.push_back(std::move(g));
    }

    // 处理只有 memory_space_cast 没有 fixpipe 的 C→V 组
    for (auto &p : memorySpaceCastByDefOp) {
        Operation *defOp = p.first;
        bool found = false;
        for (auto &g : groups) {
            if (g.buffer0CtoV && g.buffer0CtoV.getDefiningOp() == defOp) {
                found = true;
                break;
            }
        }

        if (!found) {
            TransferGroup g;
            g.ssbufferId = ssbufferIdByDefOp[defOp];
            g.fixpipeOp = nullptr;
            g.memorySpaceCastOp = p.second;
            g.buffer0CtoV = bufferValueByDefOp[defOp];
            g.hirCopyOp = nullptr;
            g.convertLayoutOp = nullptr;
            g.buffer0VtoC = Value();  // 初始化为空
            g.buffer1Created = false;
            groups.push_back(std::move(g));
        }
    }

    // =============================================================
    // V→C 传输对: hir.copy (VECTOR) <-> convert_layout (CUBE)
    // =============================================================

    // 处理有 hir.copy 的 V→C 组
    for (auto &p : hirCopyByDefOp) {
        Operation *defOp = p.first;
        TransferGroup g;
        g.ssbufferId = ssbufferIdByDefOp[defOp];
        g.hirCopyOp = p.second;
        g.buffer0VtoC = bufferValueByDefOp[defOp];
        g.convertLayoutOp = convertLayoutByDefOp.count(defOp) ?
                            convertLayoutByDefOp[defOp] : nullptr;
        g.fixpipeOp = nullptr;
        g.memorySpaceCastOp = nullptr;
        g.buffer0CtoV = Value();  // V→C 组没有 C→V buffer，初始化为空
        g.buffer1Created = false;

        groups.push_back(std::move(g));
    }

    // 处理只有 convert_layout 没有 hir.copy 的 V→C 组
    for (auto &p : convertLayoutByDefOp) {
        Operation *defOp = p.first;
        bool found = false;
        for (auto &g : groups) {
            if (g.buffer0VtoC && g.buffer0VtoC.getDefiningOp() == defOp) {
                found = true;
                break;
            }
        }

        if (!found) {
            TransferGroup g;
            g.ssbufferId = ssbufferIdByDefOp[defOp];
            g.hirCopyOp = nullptr;
            g.convertLayoutOp = p.second;
            g.buffer0VtoC = bufferValueByDefOp[defOp];
            g.fixpipeOp = nullptr;
            g.memorySpaceCastOp = nullptr;
            g.buffer0CtoV = Value();  // 初始化为空
            g.buffer1Created = false;
            groups.push_back(std::move(g));
        }
    }

    return groups;
}

// =========================================================================
// Step 2: wrapSyncOpWithCounterIf
// 将单个 void sync op 替换为:
//   scf.if (counter % c2 == 0) {
//     clone_op { static_flag_id = flag0 }
//   } else {
//     clone_op { static_flag_id = flag1 }
//   }
// counter 和 c2 均为 i32 SSA 值（已在循环体开头创建）
// =========================================================================
static void wrapSyncOpWithCounterIf(
        OpBuilder &builder, Location loc,
        Operation *syncOp,
        Value counter, Value c2,
        int64_t flag0, int64_t flag1) {

    // 在 syncOp 前插入 counter%2 判断
    builder.setInsertionPoint(syncOp);

    Type i32Type = counter.getType();
    Value c0 = builder.create<arith::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(0)).getResult();
    Value idx  = builder.create<arith::RemSIOp>(loc, counter, c2);
    Value cond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, idx, c0);

    // scf.if with no results (sync ops are pure side-effects, void return).
    // NOTE: IfOp::build with TypeRange{} calls ensureTerminator which AUTO-INSERTS
    //       scf.yield into both branches. We must NOT create another yield manually;
    //       instead, insert the cloned sync op BEFORE the existing auto-yield.
    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, /*withElse=*/true);

    // --- then branch: use flag0 ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block &thenBlock = ifOp.getThenRegion().front();
        // thenBlock = [yield(auto)]; insert clone before the yield
        builder.setInsertionPoint(&thenBlock.back());
        Operation *clone0 = syncOp->clone();
        clone0->setAttr("static_flag_id",
            builder.getIntegerAttr(builder.getI64Type(), flag0));
        builder.insert(clone0);
        // thenBlock = [clone0, yield(auto)] ✓
    }

    // --- else branch: use flag1 ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block &elseBlock = ifOp.getElseRegion().front();
        // elseBlock = [yield(auto)]; insert clone before the yield
        builder.setInsertionPoint(&elseBlock.back());
        Operation *clone1 = syncOp->clone();
        clone1->setAttr("static_flag_id",
            builder.getIntegerAttr(builder.getI64Type(), flag1));
        builder.insert(clone1);
        // elseBlock = [clone1, yield(auto)] ✓
    }

    // 原 op 已被 scf.if 替代，删除
    syncOp->erase();
}

// =========================================================================
// Step 2a: createBuffersForTransferGroup
// 在 scf.for 外部创建 N 个 buffer (memref.alloc)
// 返回创建的 buffer 列表
// =========================================================================
static SmallVector<Value>
createBuffersForTransferGroup(
    OpBuilder &builder, Location loc,
    scf::ForOp forOp,
    Type elemType, ArrayRef<int64_t> shape,
    AddressSpace addrSpace, int N = 2) {

    SmallVector<Value> buffers;

    // 在 scf.for 之前插入 alloc
    builder.setInsertionPoint(forOp);

    for (int i = 0; i < N; ++i) {
        MemRefType memrefType = MemRefType::get(shape, elemType,
            MemRefLayoutAttrInterface{},
            AddressSpaceAttr::get(builder.getContext(), addrSpace));
        auto allocOp = builder.create<memref::AllocOp>(loc, memrefType);
        buffers.push_back(allocOp.getResult());
    }

    return buffers;
}

// =========================================================================
// Step 2b: getTransferOpBufferAddressSpace
// 根据 transfer op 类型获取对应的 address space
// fixpipe 使用 UB (CUBE->VECTOR 数据)
// hir.copy 使用 CBUF/L1 (VECTOR->CUBE 数据)
// =========================================================================
static AddressSpace getTransferOpAddressSpace(Operation *op) {
    if (isFixpipeOp(op)) {
        return AddressSpace::UB;  // C→V 使用 UB
    } else if (isCopyOp(op)) {
        return AddressSpace::L1;  // V→C 使用 L1
    }
    return AddressSpace::UB;  // 默认
}

// =========================================================================
// Step 2c: wrapTransferOpWithCounterIf
// 将 transfer op (fixpipe/hir.copy) 包装在 scf.if 中
// 根据 counter % N 选择对应的 buffer
// 假设 transfer op 的 output buffer 是第 0 个 operand
// =========================================================================
static void wrapTransferOpWithCounterIf(
    OpBuilder &builder, Location loc,
    Operation *transferOp,
    Value counter, Value c2,
    ArrayRef<Value> buffers) {

    int N = buffers.size();
    if (N == 0)
        return;

    // 在 transferOp 前插入 counter%N 判断
    builder.setInsertionPoint(transferOp);

    Type i32Type = counter.getType();
    Value c0 = builder.create<arith::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(0)).getResult();
    Value idx = builder.create<arith::RemSIOp>(loc, counter, c2);
    Value cond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, idx, c0);

    // 获取 transfer op 的结果类型
    SmallVector<Type> resultTypes;
    for (auto result : transferOp->getResults())
        resultTypes.push_back(result.getType());

    auto ifOp = builder.create<scf::IfOp>(loc, resultTypes, cond, /*withElse=*/true);

    // --- then branch: 使用 buffer[0] ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block &thenBlock = ifOp.getThenRegion().front();
        builder.setInsertionPoint(&thenBlock.back());

        // 克隆 transfer op，替换 output buffer
        Operation *clone0 = transferOp->clone();
        // 假设 output buffer 是第 0 个 operand
        if (clone0->getNumOperands() > 0) {
            clone0->setOperand(0, buffers[0]);
        }
        builder.insert(clone0);

        // yield 结果
        SmallVector<Value> yieldValues;
        for (auto result : clone0->getResults())
            yieldValues.push_back(result);
        builder.create<scf::YieldOp>(loc, yieldValues);
    }

    // --- else branch: 使用 buffer[1] (N=2 的情况) ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block &elseBlock = ifOp.getElseRegion().front();
        builder.setInsertionPoint(&elseBlock.back());

        Operation *clone1 = transferOp->clone();
        if (clone1->getNumOperands() > 0) {
            clone1->setOperand(0, buffers[1]);
        }
        builder.insert(clone1);

        SmallVector<Value> yieldValues;
        for (auto result : clone1->getResults())
            yieldValues.push_back(result);
        builder.create<scf::YieldOp>(loc, yieldValues);
    }

    // 用 ifOp 的结果替换原 transfer op 的结果
    for (size_t i = 0; i < transferOp->getNumResults(); ++i) {
        transferOp->getResult(i).replaceAllUsesWith(ifOp.getResult(i));
    }

    // 删除原 op
    transferOp->erase();
}

// =========================================================================
// Step 2d: wrapTransferOpWithCounterIfSimple (简化版，无结果)
// 用于没有结果的 transfer op
// 注意: fixpipe 和 hir.copy 的输出 buffer 是 operand 1 (不是 operand 0)
// =========================================================================
static void wrapTransferOpWithCounterIfSimple(
    OpBuilder &builder, Location loc,
    Operation *transferOp,
    Value counter, Value c2,
    ArrayRef<Value> buffers) {

    int N = buffers.size();
    if (N == 0)
        return;

    // 获取 ssbuffer.block_id
    int ssbufferId = getSsbufferId(transferOp);

    builder.setInsertionPoint(transferOp);

    Type i32Type = counter.getType();
    Value c0 = builder.create<arith::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(0)).getResult();
    if (ssbufferId >= 0)
        c0.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
    Value idx = builder.create<arith::RemSIOp>(loc, counter, c2);
    if (ssbufferId >= 0)
        idx.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
    Value cond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, idx, c0);
    if (ssbufferId >= 0)
        cond.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));

    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, /*withElse=*/true);
    if (ssbufferId >= 0)
        ifOp->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));

    // --- then branch: 使用 buffer[0] ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block &thenBlock = ifOp.getThenRegion().front();
        builder.setInsertionPoint(&thenBlock.back());
        Operation *clone0 = transferOp->clone();
        // fixpipe/hir.copy 的输出 buffer 是 operand 1
        if (clone0->getNumOperands() > 1)
            clone0->setOperand(1, buffers[0]);
        builder.insert(clone0);
    }

    // --- else branch: 使用 buffer[1] ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block &elseBlock = ifOp.getElseRegion().front();
        builder.setInsertionPoint(&elseBlock.back());
        Operation *clone1 = transferOp->clone();
        if (clone1->getNumOperands() > 1)
            clone1->setOperand(1, buffers[1]);
        builder.insert(clone1);
    }

    transferOp->erase();
}

// =========================================================================
// Step 2e: wrapTransferOpWithSync (带同步点的传输包装)
// 在 scf.if 内部：
//   1. sync_block_wait (等待对方用完buffer)
//   2. transfer op (fixpipe/hir.copy)
//   3. sync_block_set (通知对方可以用buffer)
// 使用不同的 flag_id 来区分双缓冲的两个阶段
// =========================================================================
static void wrapTransferOpWithSync(
    OpBuilder &builder, Location loc,
    Operation *transferOp,
    Value counter, Value c2,
    ArrayRef<Value> buffers,
    int64_t flagId0, int64_t flagId1,
    bool isCtoV) {
    // isCtoV: true 表示 C→V (fixpipe), false 表示 V→C (hir.copy)

    int N = buffers.size();
    if (N == 0)
        return;

    MLIRContext *ctx = builder.getContext();

    // 获取 ssbuffer.block_id
    int ssbufferId = getSsbufferId(transferOp);

    builder.setInsertionPoint(transferOp);

    Type i32Type = counter.getType();
    Value c0 = builder.create<arith::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(0)).getResult();
    if (ssbufferId >= 0)
        c0.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
    Value idx = builder.create<arith::RemSIOp>(loc, counter, c2);
    if (ssbufferId >= 0)
        idx.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
    Value cond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, idx, c0);
    if (ssbufferId >= 0)
        cond.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));

    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, /*withElse=*/true);
    if (ssbufferId >= 0)
        ifOp->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));

    // 准备 sync 操作所需的参数
    // C→V: CUBE set on PIPE_FIX → VECTOR wait on PIPE_V
    // V→C: VECTOR set on PIPE_FIX → CUBE wait on PIPE_V
    auto setPipe = PipeAttr::get(ctx, hivm::PIPE::PIPE_FIX);
    auto waitPipe = PipeAttr::get(ctx, hivm::PIPE::PIPE_V);

    // --- then branch: 使用 buffer[0], flagId0 ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block &thenBlock = ifOp.getThenRegion().front();
        builder.setInsertionPoint(&thenBlock.back());

        // 1. sync_block_wait: 等待对方用完 buffer[0]
        // C→V: VECTOR waits; V→C: CUBE waits
        auto waitCoreType = isCtoV ? hivm::TCoreType::VECTOR : hivm::TCoreType::CUBE;
        auto waitCoreAttr = hivm::TCoreTypeAttr::get(ctx, waitCoreType);
        auto waitOp0 = builder.create<hivm::SyncBlockWaitOp>(
            loc, waitCoreAttr, setPipe, waitPipe,
            builder.getI64IntegerAttr(flagId0));
        if (ssbufferId >= 0)
            waitOp0->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
        (void)waitOp0;

        // 2. transfer op
        Operation *clone0 = transferOp->clone();
        if (clone0->getNumOperands() > 1)
            clone0->setOperand(1, buffers[0]);
        builder.insert(clone0);

        // 3. sync_block_set: 通知对方 buffer[0] 已准备好
        // C→V: CUBE sets; V→C: VECTOR sets
        auto setCoreType = isCtoV ? hivm::TCoreType::CUBE : hivm::TCoreType::VECTOR;
        auto setCoreAttr = hivm::TCoreTypeAttr::get(ctx, setCoreType);
        auto setOp0 = builder.create<hivm::SyncBlockSetOp>(
            loc, setCoreAttr, setPipe, waitPipe,
            builder.getI64IntegerAttr(flagId0));
        if (ssbufferId >= 0)
            setOp0->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
        (void)setOp0;
    }

    // --- else branch: 使用 buffer[1], flagId1 ---
    {
        OpBuilder::InsertionGuard guard(builder);
        Block &elseBlock = ifOp.getElseRegion().front();
        builder.setInsertionPoint(&elseBlock.back());

        // 1. sync_block_wait: 等待对方用完 buffer[1]
        auto waitCoreType = isCtoV ? hivm::TCoreType::VECTOR : hivm::TCoreType::CUBE;
        auto waitCoreAttr = hivm::TCoreTypeAttr::get(ctx, waitCoreType);
        auto waitOp1 = builder.create<hivm::SyncBlockWaitOp>(
            loc, waitCoreAttr, setPipe, waitPipe,
            builder.getI64IntegerAttr(flagId1));
        if (ssbufferId >= 0)
            waitOp1->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
        (void)waitOp1;

        // 2. transfer op
        Operation *clone1 = transferOp->clone();
        if (clone1->getNumOperands() > 1)
            clone1->setOperand(1, buffers[1]);
        builder.insert(clone1);

        // 3. sync_block_set: 通知对方 buffer[1] 已准备好
        auto setCoreType = isCtoV ? hivm::TCoreType::CUBE : hivm::TCoreType::VECTOR;
        auto setCoreAttr = hivm::TCoreTypeAttr::get(ctx, setCoreType);
        auto setOp1 = builder.create<hivm::SyncBlockSetOp>(
            loc, setCoreAttr, setPipe, waitPipe,
            builder.getI64IntegerAttr(flagId1));
        if (ssbufferId >= 0)
            setOp1->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
        (void)setOp1;
    }

    // 删除原始的 sync ops（在删除 transfer op 之前）
    eraseOriginalSyncOpsInScope(transferOp);

    transferOp->erase();
}

// =========================================================================
// Step 3: addCVMultiBuffer — 处理 CV 间多 buffer 的主协调函数
//
// 通过 buffer 识别传输组，每组包含：
//   C→V: fixpipe (CUBE 发端) + memory_space_cast (VECTOR 收端)
//   V→C: hir.copy (VECTOR 发端) + convert_layout (CUBE 收端)
//
// 双 buffer: 创建新 buffer (g.buffer1)，与原始 buffer 配对
// counter 自增: 在 V→C 收端 (convert_layout 侧) 添加 counter = counter + 1
// =========================================================================
static void addCVMultiBuffer(
    SmallVector<TransferGroup> &groups,
    ModuleOp module,
    int N = 2) {

    OpBuilder builder(module.getContext());

    // 初始化 FlagIdManager，扫描已存在的 flag_id
    mlir::triton::FlagIdManager flagIdMgr(module);

    for (TransferGroup &g : groups) {
        // 获取 buffer 类型信息 - 优先使用 C→V 的 buffer
        Value originalBuffer;
        if (g.buffer0CtoV) {
            originalBuffer = g.buffer0CtoV;
        } else if (g.buffer0VtoC) {
            originalBuffer = g.buffer0VtoC;
        } else {
            // 没有有效的 buffer
            continue;
        }

        auto bufferType = dyn_cast<MemRefType>(originalBuffer.getType());
        if (!bufferType)
            continue;

        Type elemType = bufferType.getElementType();
        auto shape = llvm::to_vector(bufferType.getShape());
        auto addrSpaceAttr = bufferType.getMemorySpace();

        Location loc = originalBuffer.getDefiningOp() ?
                       originalBuffer.getDefiningOp()->getLoc() :
                       builder.getUnknownLoc();
        Type i32Type = builder.getI32Type();

        // =============================================================
        // 创建第二个 buffer (buffer1)
        // 在外层 scope 创建，让两个 scope 都能访问
        // =============================================================
        Operation *insertOp = g.fixpipeOp ?
                              g.fixpipeOp :
                              (g.memorySpaceCastOp ?
                               g.memorySpaceCastOp :
                               (g.hirCopyOp ?
                                g.hirCopyOp :
                                g.convertLayoutOp));
        if (!insertOp)
            continue;

        auto [outerBlock, outerIt] = getOuterScopeInsertPointBeforeInnerScopes(insertOp);
        if (!outerBlock)
            continue;

        builder.setInsertionPoint(outerBlock, outerIt);

        MemRefType memrefType = MemRefType::get(shape, elemType,
            MemRefLayoutAttrInterface{},
            addrSpaceAttr);
        auto allocOp = builder.create<memref::AllocOp>(loc, memrefType);
        // 添加 ssbuffer.block_id 属性
        if (g.ssbufferId >= 0)
            allocOp->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(g.ssbufferId));
        g.buffer1 = allocOp.getResult();
        g.buffer1Created = true;

        // 创建 cN 常量
        Value cN = builder.create<arith::ConstantOp>(
            loc, i32Type, builder.getI32IntegerAttr(N)).getResult();

        // 双 buffer 列表: [原始 buffer, buffer1]
        SmallVector<Value, 2> buffers;

        // =============================================================
        // 处理 C→V 传输 (fixpipe + memory_space_cast)
        // =============================================================
        if (g.fixpipeOp || g.memorySpaceCastOp) {
            // C→V 使用 buffer0CtoV
            Value buf0 = g.buffer0CtoV ? g.buffer0CtoV : g.buffer1;
            buffers = {buf0, g.buffer1};

            // 分配 flag_id (C→V 用一对)
            // flagId0 复用原有的 flag，flagId1 通过 FlagIdManager 分配
            int64_t flagId0 = -1;
            int64_t flagId1 = flagIdMgr.acquireId();

            // 查找 CUBE 侧原有的 sync_block_wait (C→V 方向，wait for VECTOR)
            Operation *originalSyncWait = findOriginalSyncOpInScope(
                g.fixpipeOp, /*isWait=*/true, /*isCtoV=*/true, flagId0);

            // 如果找不到原有的 sync op，fallback 到新分配
            if (flagId0 < 0 || originalSyncWait == nullptr) {
                flagId0 = flagIdMgr.acquireId();
            }

            // ---------------------------------------------------------
            // 包装 fixpipe (CUBE 发端)
            // counter%2==0 用 buffer0, counter%2==1 用 buffer1
            // ---------------------------------------------------------
            if (g.fixpipeOp) {
                scf::ForOp cubeFor = findEnclosingForOp(g.fixpipeOp);
                if (cubeFor) {
                    Block *cubeBody = cubeFor.getBody();
                    // 在 for body 开头创建 counter
                    builder.setInsertionPointToStart(cubeBody);
                    Value loopVar = cubeBody->getArgument(0);
                    Value stepVal = cubeFor.getStep();

                    // 获取 ssbufferId 在创建 counter 之前
                    int ssbufferId = getSsbufferId(g.fixpipeOp);

                    Value counter = builder.create<arith::DivSIOp>(
                        loc, loopVar, stepVal).getResult();
                    if (ssbufferId >= 0)
                        counter.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));

                    // 包装 fixpipe (C→V: isCtoV=true)
                    wrapTransferOpWithSync(
                        builder, cubeFor.getLoc(), g.fixpipeOp, counter, cN, buffers,
                        flagId0, flagId1, /*isCtoV=*/true);
                }
            }

            // ---------------------------------------------------------
            // 包装 memory_space_cast (VECTOR 收端)
            // ---------------------------------------------------------
            if (g.memorySpaceCastOp) {
                scf::ForOp vectorFor = findEnclosingForOp(g.memorySpaceCastOp);
                if (vectorFor) {
                    Block *vectorBody = vectorFor.getBody();
                    // 在 for body 开头创建 counter
                    builder.setInsertionPointToStart(vectorBody);
                    Value loopVar = vectorBody->getArgument(0);
                    Value stepVal = vectorFor.getStep();

                    // 获取 ssbufferId 在创建 counter 之前
                    int ssbufferId = getSsbufferId(g.memorySpaceCastOp);

                    Value counter = builder.create<arith::DivSIOp>(
                        loc, loopVar, stepVal).getResult();
                    if (ssbufferId >= 0)
                        counter.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));

                    // 预先计算条件 (counter % c2 == 0)
                    Type i32Type = counter.getType();
                    Value c0 = builder.create<arith::ConstantOp>(
                        loc, i32Type, builder.getI32IntegerAttr(0)).getResult();
                    if (ssbufferId >= 0)
                        c0.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
                    Value idx = builder.create<arith::RemSIOp>(loc, counter, cN);
                    if (ssbufferId >= 0)
                        idx.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
                    Value cond = builder.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::eq, idx, c0);
                    if (ssbufferId >= 0)
                        cond.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));

                    // 1. Insert sync_block_wait before memory_space_cast (使用预计算的条件)
                    wrapReceiveOpWithWaitIf(
                        builder, vectorFor.getLoc(), g.memorySpaceCastOp, counter, cN,
                        flagId0, flagId1, /*isCtoV=*/true, cond);

                    // 2. Wrap the receive operation with counter-based selection (使用预计算的条件)
                    // 注意：g.memorySpaceCastOp 在 wrapReceiveOpWithWaitIf 之后已经被erase了，
                    // 所以这里需要重新获取。但因为我们已经在 wait 之前处理，这里需要用不同的方式。
                    // 实际上 wrapReceiveOpWithWaitIf 不会 erase receiveOp，只有 wrapReceiveOpWithReceiveIf 会
                    scf::IfOp receiveIfOp = wrapReceiveOpWithReceiveIf(
                        builder, vectorFor.getLoc(), g.memorySpaceCastOp, counter, cN, buffers, cond);

                    // 3. Insert sync_block_set after the receive operation (使用预计算的条件)
                    if (receiveIfOp && receiveIfOp.getNumResults() > 0) {
                        wrapReceiveOpWithSetIf(
                            builder, vectorFor.getLoc(), receiveIfOp.getResult(0), counter, cN,
                            flagId0, flagId1, /*isCtoV=*/true, ssbufferId, cond);
                    }
                }
            }
        }

        // =============================================================
        // 处理 V→C 传输 (hir.copy + convert_layout)
        // =============================================================
        if (g.hirCopyOp || g.convertLayoutOp) {
            // V→C 使用 buffer0VtoC
            Value buf0 = g.buffer0VtoC ? g.buffer0VtoC : g.buffer1;
            buffers = {buf0, g.buffer1};

            // 分配 flag_id (V→C 用另一对，与 C→V 区分)
            // flagId0 复用原有的 flag，flagId1 通过 FlagIdManager 分配
            int64_t flagId0 = -1;
            int64_t flagId1 = flagIdMgr.acquireId();

            // 查找 VECTOR 侧原有的 sync_block_wait (V→C 方向，wait for CUBE)
            Operation *originalSyncWait = findOriginalSyncOpInScope(
                g.hirCopyOp, /*isWait=*/true, /*isCtoV=*/false, flagId0);

            // 如果找不到原有的 sync op，fallback 到新分配
            if (flagId0 < 0 || originalSyncWait == nullptr) {
                flagId0 = flagIdMgr.acquireId();
            }

            // ---------------------------------------------------------
            // 包装 hir.copy (VECTOR 发端)
            // counter%2==0 用 buffer0, counter%2==1 用 buffer1
            // ---------------------------------------------------------
            if (g.hirCopyOp) {
                scf::ForOp vectorFor = findEnclosingForOp(g.hirCopyOp);
                if (vectorFor) {
                    Block *vectorBody = vectorFor.getBody();
                    // 在 for body 开头创建 counter
                    builder.setInsertionPointToStart(vectorBody);
                    Value loopVar = vectorBody->getArgument(0);
                    Value stepVal = vectorFor.getStep();

                    // 获取 ssbufferId 在创建 counter 之前
                    int ssbufferId = getSsbufferId(g.hirCopyOp);

                    Value counter = builder.create<arith::DivSIOp>(
                        loc, loopVar, stepVal).getResult();
                    if (ssbufferId >= 0)
                        counter.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));

                    // 包装 hir.copy (V→C: isCtoV=false)
                    wrapTransferOpWithSync(
                        builder, vectorFor.getLoc(), g.hirCopyOp, counter, cN, buffers,
                        flagId0, flagId1, /*isCtoV=*/false);
                }
            }

            // ---------------------------------------------------------
            // 包装 convert_layout (CUBE 收端)
            // ---------------------------------------------------------
            if (g.convertLayoutOp) {
                scf::ForOp cubeFor = findEnclosingForOp(g.convertLayoutOp);
                if (cubeFor) {
                    Block *cubeBody = cubeFor.getBody();
                    // 在 for body 开头创建 counter
                    builder.setInsertionPointToStart(cubeBody);
                    Value loopVar = cubeBody->getArgument(0);
                    Value stepVal = cubeFor.getStep();

                    // 获取 ssbufferId 在创建 counter 之前
                    int ssbufferId = getSsbufferId(g.convertLayoutOp);

                    Value counter = builder.create<arith::DivSIOp>(
                        loc, loopVar, stepVal).getResult();
                    if (ssbufferId >= 0)
                        counter.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));

                    // 预先计算条件 (counter % c2 == 0)
                    Type i32Type = counter.getType();
                    Value c0 = builder.create<arith::ConstantOp>(
                        loc, i32Type, builder.getI32IntegerAttr(0)).getResult();
                    if (ssbufferId >= 0)
                        c0.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
                    Value idx = builder.create<arith::RemSIOp>(loc, counter, cN);
                    if (ssbufferId >= 0)
                        idx.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
                    Value cond = builder.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::eq, idx, c0);
                    if (ssbufferId >= 0)
                        cond.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));

                    // 1. Insert sync_block_wait before convert_layout (使用预计算的条件)
                    wrapReceiveOpWithWaitIf(
                        builder, cubeFor.getLoc(), g.convertLayoutOp, counter, cN,
                        flagId0, flagId1, /*isCtoV=*/false, cond);

                    // 2. Wrap the receive operation with counter-based selection (使用预计算的条件)
                    scf::IfOp receiveIfOp = wrapReceiveOpWithReceiveIf(
                        builder, cubeFor.getLoc(), g.convertLayoutOp, counter, cN, buffers, cond);

                    // 3. Insert sync_block_set after the receive operation (使用预计算的条件)
                    if (receiveIfOp && receiveIfOp.getNumResults() > 0) {
                        wrapReceiveOpWithSetIf(
                            builder, cubeFor.getLoc(), receiveIfOp.getResult(0), counter, cN,
                            flagId0, flagId1, /*isCtoV=*/false, ssbufferId, cond);
                    }

                    // 4. 在 CUBE 收端添加 counter 自增 (V→C 方向)
                    if (receiveIfOp) {
                        builder.setInsertionPointAfter(receiveIfOp);
                        Value one = builder.create<arith::ConstantOp>(
                            loc, i32Type, builder.getI32IntegerAttr(1)).getResult();
                        if (ssbufferId >= 0)
                            one.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
                        Value counterNext = builder.create<arith::AddIOp>(loc, counter, one).getResult();
                        if (ssbufferId >= 0)
                            counterNext.getDefiningOp()->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(ssbufferId));
                        (void)counterNext;
                    }
                }
            }
        }
    }
}

// =========================================================================
// Pass 主体
// =========================================================================
struct OuterMultiBufferPass
    : public mlir::triton::impl::OuterMultiBufferBase<OuterMultiBufferPass> {
    void runOnOperation() override;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect, memref::MemRefDialect,
                        hivm::HIVMDialect, scope::ScopeDialect,
                        bufferization::BufferizationDialect>();
    }
};

void OuterMultiBufferPass::runOnOperation() {
    auto module = getOperation();

    // 清空 crossCoreDependentMap (防止多次 runOnOperation 调用时残留)
    crossCoreDependentMap.clear();

    // Step 1: 找到 CUBE↔VECTOR 传输组（按 ssbuffer.id 匹配）
    auto groups = findTransferGroups(module);
    if (groups.empty())
        return;

    // Step 2: 处理 CV 间多 buffer (fixpipe/hir.copy/memory_space_cast/convert_layout)
    // 注: sync 点处理暂不启用 (flag 分配逻辑有问题)
    addCVMultiBuffer(groups, module, /*N=*/2);

    // Step 3: 重排操作，将相同 ssbuffer.block_id 的操作聚集到一起
    // 注意：由于存在循环依赖（如 scf.for 的 iter_args 依赖），暂时禁用
    // TODO: 需要更好的算法处理 region 嵌套操作和循环依赖
    // reorderOpsBySsbufferIdTopo(module);

    // 打印 crossCoreDependentMap 用于调试
    llvm::errs() << "=== crossCoreDependentMap ===\n";
    OpPrintingFlags flags;
    for (auto &kv : crossCoreDependentMap) {
        kv.getFirst().printAsOperand(llvm::errs() << "  ", flags);
        llvm::errs() << " -> {";
        for (size_t i = 0; i < kv.getSecond().size(); ++i) {
            if (i > 0) llvm::errs() << ", ";
            kv.getSecond()[i].printAsOperand(llvm::errs(), flags);
        }
        llvm::errs() << "}\n";
    }
    // 将 crossCoreDependentMap 数据写入全局单例 BufferRelationAnalysis，供下游 Pass 使用
    auto &globalRel = mlir::triton::getGlobalBufferRelation();
    for (auto &entry : crossCoreDependentMap) {
        globalRel.addCrossCoreRelation(entry.first, entry.second);
    }
    llvm::errs() << "[OuterMultiBuffer] 已将 crossCoreDependentMap 写入全局单例, size="
                 << globalRel.getCrossCoreRelations().size() << "\n";

}

} // anonymous namespace

// =========================================================================
// 工厂函数（供 Pass 框架和测试工具调用）
// =========================================================================
std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createOuterMultiBufferPass() {
    return std::make_unique<OuterMultiBufferPass>();
}
