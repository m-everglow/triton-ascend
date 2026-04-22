/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * 调试开关:
 *   DEBUG_MODE = 0 : 只输出 IR（无调试信息）
 *   DEBUG_MODE = 1 : 只输出调试信息（不打印 IR） IR仍输出但可重定向过滤
 *   DEBUG_MODE = 2 : 同时输出 IR + 调试信息
 *
 * 修改后需重新编译: bash /home/zdl/triton-ascend/build_triton_adapter_opt.sh
 */
#define DEBUG_MODE 1

#define GEN_PASS_DEF_MULTIBUFFER
#include "TritonAffinityOpt/Passes.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

#include "Utils/Utils.h"
#include "TritonAffinityOpt/BufferRelationAnalysis.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>


namespace mlir {
namespace triton {
#include "ascend/include/TritonAffinityOpt/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {

struct BlockInfo {
    Value blockId;
    SmallVector<Operation *> ops;
};

static int getSsbufferId(Operation *op) {
    // 真实 IR 使用 ssbuffer.block_id，上游旧 IR 使用 ssbuffer.id
    if (auto idAttr = op->getAttrOfType<IntegerAttr>("ssbuffer.block_id"))
        return idAttr.getInt();
    if (auto idAttr = op->getAttrOfType<IntegerAttr>("ssbuffer.id"))
        return idAttr.getInt();
    return -1;
}

static void collectNestedOps(Block *block, SmallVector<Operation *> &ops) {
    for (auto &op : *block) {
        ops.push_back(&op);
        for (auto &region : op.getRegions())
            for (auto &innerBlock : region)
                collectNestedOps(&innerBlock, ops);
    }
}

// Step 1: 收集 block 信息，按 ssbuffer.id 分组
static DenseMap<Value, SmallVector<Value>>
collectBlockInfo(scf::ForOp forOp, DenseMap<Value, BlockInfo> &blocks) {
    DenseMap<Value, SmallVector<Value>> depValueMap;
    Block *body = forOp.getBody();
    if (!body) return depValueMap;

    llvm::MapVector<Value, Operation *> opsByValue;
    SmallVector<Operation *> allOps;
    collectNestedOps(body, allOps);

    for (Operation *op : allOps) {
        int id = getSsbufferId(op);
        if (id < 0) continue;
        for (auto res : op->getResults())
            opsByValue[res] = op;
    }
    if (opsByValue.empty()) return depValueMap;

    llvm::MapVector<int, SmallVector<Operation *>> opsById;
    for (auto &p : opsByValue) {
        int id = getSsbufferId(p.second);
        if (id >= 0) opsById[id].push_back(p.second);
    }

    DenseMap<Value, int> outputToBlockId;
    for (auto &p : opsById)
        for (Operation *op : p.second)
            for (auto res : op->getResults())
                outputToBlockId[res] = p.first;

    for (auto &p : opsById) {
        Value groupKey = p.second.front()->getResult(0);
        BlockInfo bi;
        bi.blockId = groupKey;
        bi.ops = p.second;
        for (Operation *op : bi.ops) {
            // 只收集跨 id 的依赖：operands 来自更小 id 的 op 输出
            for (Value operand : op->getOperands()) {
                if (auto barg = dyn_cast<BlockArgument>(operand)) {
                    if (barg.getOwner() == body &&
                        !llvm::is_contained(depValueMap[groupKey], barg))
                        depValueMap[groupKey].push_back(barg);
                    continue;
                }
                if (outputToBlockId.count(operand) &&
                    outputToBlockId[operand] < p.first &&
                    !llvm::is_contained(depValueMap[groupKey], operand))
                    depValueMap[groupKey].push_back(operand);
            }
        }
        blocks[groupKey] = bi;
    }
    return depValueMap;
}

static DenseMap<Value, SmallVector<Operation *>>
buildDepUserMap(DenseMap<Value, BlockInfo> &blocks) {
    DenseMap<Value, SmallVector<Operation *>> depUserMap;
    for (auto &p : blocks)
        for (Operation *op : p.second.ops)
            for (Value operand : op->getOperands())
                depUserMap[operand].push_back(op);
    return depUserMap;
}

// 收集所有需要 buffer 的 depVal（每个 depVal 独立分配一对 buffer）
static SmallVector<Value>
collectBufferValues(DenseMap<Value, SmallVector<Value>> &depValueMap) {
    SmallVector<Value> valueList;
    SmallVector<void *> seenPtrs;
    for (auto &p : depValueMap)
        for (Value depVal : p.second)
            if (depVal.getDefiningOp() &&
                !llvm::is_contained(seenPtrs, depVal.getAsOpaquePointer())) {
                seenPtrs.push_back(depVal.getAsOpaquePointer());
                auto shapedType = dyn_cast<ShapedType>(depVal.getType());
                if (!shapedType) continue;
                valueList.push_back(depVal);
            }
    return valueList;
}

// 在 for 循环之前（scope 内，for 外）一次性插入所有 buffer allocation
// 每个 depVal 独立分配一对 buffer，避免读写冲突
// 返回 bufferMap（Value -> [(tensor, memref) pairs]）
static DenseMap<Value, SmallVector<std::pair<Value, Value>>>
insertBuffersBeforeFor(scf::ForOp forOp,
                       SmallVector<Value> &valueList,
                       OpBuilder &builder) {
    DenseMap<Value, SmallVector<std::pair<Value, Value>>> bufferMap;
    Block *parentBlock = forOp->getBlock();
    OpBuilder ib(builder.getContext());
    ib.setInsertionPoint(parentBlock, forOp->getIterator());

    constexpr int bufNum = 2;
    for (Value depVal : valueList) {
        ShapedType shapedType = cast<ShapedType>(depVal.getType());
        Type elemType = shapedType.getElementType();
        AddressSpace addrSpace = AddressSpace::UB;

        SmallVector<std::pair<Value, Value>> buffers; // (tensor, memref) pairs
        for (int i = 0; i < bufNum; ++i) {
            MemRefType memrefType = MemRefType::get(shapedType.getShape(), elemType,
                                                    MemRefLayoutAttrInterface{},
                                                    AddressSpaceAttr::get(ib.getContext(), addrSpace));
            auto allocOp = ib.create<memref::AllocOp>(forOp.getLoc(), memrefType);
            auto genericType = MemRefType::get(shapedType.getShape(), elemType,
                                                MemRefLayoutAttrInterface{}, 0u);
            auto casted = ib.create<memref::MemorySpaceCastOp>(forOp.getLoc(), genericType, allocOp.getResult());
            auto tensorOutType = RankedTensorType::get(shapedType.getShape(), elemType);
            auto toTensor = ib.create<bufferization::ToTensorOp>(forOp.getLoc(), tensorOutType, casted.getResult());
            // 存 casted（generic memref），供循环内 TensorStoreOp / ToTensorOp 使用
            buffers.push_back({toTensor.getResult(), casted.getResult()});
        }
        bufferMap[depVal] = buffers;
    }
    return bufferMap;
}

// 从 for 循环 IV 计算当前迭代号，返回 i32
// 支持 IV 为 index、i32、i64 等整数类型
// 同时通过 newOps 收集所有新创建的 op（SubIOp, DivUIOp, CastOp 等）
static Value getIterCount(OpBuilder &builder, scf::ForOp forOp, Location loc,
                          SmallVector<Operation *> *newOps) {
    auto i32Type = builder.getI32Type();
    Value iv   = forOp.getInductionVar();
    Value lb   = forOp.getLowerBound();
    Value step = forOp.getStep();
    Type ivType = iv.getType();

    // (iv - lb) / step  in native IV type
    Value diff    = builder.create<arith::SubIOp>(loc, iv, lb);
    Value iterIdx = builder.create<arith::DivUIOp>(loc, diff, step);
    newOps->push_back(diff.getDefiningOp());
    newOps->push_back(iterIdx.getDefiningOp());

    // Cast to i32 if needed
    if (ivType == i32Type)
        return iterIdx; // already i32
    if (ivType.isIndex()) {
        Value result = builder.create<arith::IndexCastOp>(loc, i32Type, iterIdx);
        newOps->push_back(result.getDefiningOp());
        return result;
    }
    if (auto intType = dyn_cast<IntegerType>(ivType)) {
        if (intType.getWidth() < 32) {
            Value result = builder.create<arith::ExtSIOp>(loc, i32Type, iterIdx);
            newOps->push_back(result.getDefiningOp());
            return result;
        }
        if (intType.getWidth() > 32) {
            Value result = builder.create<arith::TruncIOp>(loc, i32Type, iterIdx);
            newOps->push_back(result.getDefiningOp());
            return result;
        }
    }
    // Fallback: index_cast (works for index-like types)
    Value result = builder.create<arith::IndexCastOp>(loc, i32Type, iterIdx);
    newOps->push_back(result.getDefiningOp());
    return result;
}

// Producer 双缓冲写入逻辑（紧跟 depDefinedOp 之后插入）
// 使用 build(TypeRange, cond, addThenBlock, addElseBlock) 避免 auto-yield 问题
// 返回所有新创建的 op，供调用者批量打 ssbuffer.id 标签
static SmallVector<Operation *>
insertProducerLogic(OpBuilder &builder, Value depVal,
                   SmallVector<std::pair<Value, Value>> &buffers,
                   scf::ForOp forOp) {
    SmallVector<Operation *> newOps;
    int N = buffers.size();
    Location loc = depVal.getLoc();
    auto i32Type = builder.getI32Type();

    Value iterCount = getIterCount(builder, forOp, loc, &newOps);

    Value Nval = builder.create<arith::ConstantIntOp>(loc, N, i32Type);
    Value bufIdx = builder.create<arith::RemSIOp>(loc, iterCount, Nval);
    newOps.push_back(Nval.getDefiningOp());
    newOps.push_back(bufIdx.getDefiningOp());

    for (int i = 0; i < N; ++i) {
        Value iVal = builder.create<arith::ConstantIntOp>(loc, i, i32Type);
        Value cond = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, bufIdx, iVal);
        auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond,
                                               /*addThenBlock=*/true, /*addElseBlock=*/false);
        newOps.push_back(iVal.getDefiningOp());
        newOps.push_back(cond.getDefiningOp());
        newOps.push_back(ifOp);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        auto matOp = builder.create<bufferization::MaterializeInDestinationOp>(
            loc, mlir::Type{}, depVal, buffers[i].second,
            /*restrict=*/mlir::UnitAttr{}, /*writable=*/builder.getUnitAttr());
        newOps.push_back(matOp);
        builder.create<scf::YieldOp>(loc); // 手动插入 yield
        builder.setInsertionPointAfter(ifOp); // 恢复到 ifOp 之后
    }
    return newOps;
}

// Consumer 双缓冲读取逻辑（在 depUser 之前插入）
// readIdx = (iterCount + N - 1) % N；iterCount==0 时 readIdx=N-1（warmup）
// 返回新创建的 op（含 scf.if），供调用者批量打 ssbuffer.id 标签
static SmallVector<Operation *>
insertConsumerLogic(OpBuilder &builder,
                    SmallVector<std::pair<Value, Value>> &buffers,
                    scf::ForOp forOp,
                    SmallVector<Operation *> &outIfOps) {
    SmallVector<Operation *> newOps;
    int N = buffers.size();
    Location loc = builder.getInsertionPoint()->getLoc();
    auto i32Type = builder.getI32Type();

    Value iterCount = getIterCount(builder, forOp, loc, &newOps);

    Value Nval = builder.create<arith::ConstantIntOp>(loc, N, i32Type);
    Value Nm1  = builder.create<arith::ConstantIntOp>(loc, N - 1, i32Type);
    Value adj  = builder.create<arith::AddIOp>(loc, iterCount, Nm1);
    Value readIdx = builder.create<arith::RemSIOp>(loc, adj, Nval);
    newOps.push_back(Nval.getDefiningOp());
    newOps.push_back(Nm1.getDefiningOp());
    newOps.push_back(adj.getDefiningOp());
    newOps.push_back(readIdx.getDefiningOp());

    auto tensorType = mlir::cast<RankedTensorType>(buffers[0].first.getType());
    SmallVector<Type> resultTypes{tensorType};

    if (N == 2) {
        Value zero   = builder.create<arith::ConstantIntOp>(loc, 0, i32Type);
        Value isIdx0 = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, readIdx, zero);
        auto ifOp = builder.create<scf::IfOp>(loc, resultTypes, isIdx0,
                                               /*addThenBlock=*/true, /*addElseBlock=*/true);
        newOps.push_back(zero.getDefiningOp());
        newOps.push_back(isIdx0.getDefiningOp());
        newOps.push_back(ifOp);
        outIfOps.push_back(ifOp);
        // then block: load buf[0]
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        auto t0 = builder.create<bufferization::ToTensorOp>(loc, tensorType, buffers[0].second);
        newOps.push_back(t0);
        builder.create<scf::YieldOp>(loc, t0.getResult());
        // else block: load buf[1]
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        auto t1 = builder.create<bufferization::ToTensorOp>(loc, tensorType, buffers[1].second);
        newOps.push_back(t1);
        builder.create<scf::YieldOp>(loc, t1.getResult());
        builder.setInsertionPointAfter(ifOp);
        return newOps;
    }

    // N>2: chain if-else
    Value result = buffers[N - 1].first; // fallback: last buf's initial tensor
    for (int i = 0; i < N - 1; ++i) {
        Value iVal = builder.create<arith::ConstantIntOp>(loc, i, i32Type);
        Value cond = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, readIdx, iVal);
        auto ifOp = builder.create<scf::IfOp>(loc, resultTypes, cond,
                                               /*addThenBlock=*/true, /*addElseBlock=*/true);
        newOps.push_back(iVal.getDefiningOp());
        newOps.push_back(cond.getDefiningOp());
        newOps.push_back(ifOp);
        outIfOps.push_back(ifOp);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        auto ti = builder.create<bufferization::ToTensorOp>(loc, tensorType, buffers[i].second);
        newOps.push_back(ti);
        builder.create<scf::YieldOp>(loc, ti.getResult());
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        builder.create<scf::YieldOp>(loc, result);
        builder.setInsertionPointAfter(ifOp);
        result = ifOp.getResult(0);
    }
    return newOps;
}

// 供下游 Pass 使用的数据结构
//   Key: scf::ForOp — 插入 double buffer 的那个 for 循环
//   Inner Key: Value — selectedBuffer（scf.if 的结果，consumer 替换后使用的值）
//   Value: SmallVector<Value> — memref 列表（如 {memspacecast, memspacecast_1}）
using ForOpSelectedBufferMap =
    DenseMap<scf::ForOp, DenseMap<Value, SmallVector<Value>>>;

// 工具函数：为 newOps 中所有 op 批量设置 ssbuffer.block_id 属性
static void addBlockAttrForOps(SmallVector<Operation *> &newOps, int blockId,
                                OpBuilder &builder) {
    auto attr = builder.getI32IntegerAttr(blockId);
    for (auto *op : newOps)
        op->setAttr("ssbuffer.block_id", attr);
}

// Step 3: 主协调函数
// buffer 已在 for 循环之前按 depVal 独立分配好，这里插入 producer 写入 / consumer 读取逻辑
// Producer 侧新 op 打 producerId，Consumer 侧新 op 打 consumer 的 userBlockId
static void addMultiBuffCaculate(scf::ForOp forOp,
                     DenseMap<Value, SmallVector<Value>> &depValueMap,
                     DenseMap<Value, SmallVector<std::pair<Value, Value>>> &bufferMap,
                     DenseMap<Value, BlockInfo> &blocks,
                     DenseMap<Value, SmallVector<Operation *>> &depUserMap,
                     ForOpSelectedBufferMap &forOpBufferMap) {
    // 全局去重：每个 depVal（按指针唯一性）只处理一次
    SmallVector<void *> seenValuePtrs;
    OpBuilder globalBuilder(forOp.getContext());

    for (auto &p : blocks) {
        Value blockKey = p.first;
        auto depIt = depValueMap.find(blockKey);
        if (depIt == depValueMap.end()) continue;

        SmallVector<Value> &depValues = depIt->second;

        for (Value depVal : depValues) {
            // 按 Value 指针去重——同一 Value 出现在多个 block group 时只处理第一次
            if (llvm::is_contained(seenValuePtrs, depVal.getAsOpaquePointer())) continue;
            seenValuePtrs.push_back(depVal.getAsOpaquePointer());

            // 跳过 block argument（iter_args），它们不是 producer 定义的值
            if (isa<BlockArgument>(depVal)) continue;

            Operation *depDefinedOp = depVal.getDefiningOp();
            if (!depDefinedOp) continue;

            auto userIt = depUserMap.find(depVal);
            if (userIt == depUserMap.end()) continue;

            auto shapedType = dyn_cast<ShapedType>(depVal.getType());
            if (!shapedType) continue;
            // 每个 depVal 有独立的 buffer pair（避免同一 block 内多个 producer 冲突）
            auto bufferIt = bufferMap.find(depVal);
            if (bufferIt == bufferMap.end()) continue;

            SmallVector<std::pair<Value, Value>> &buffers = bufferIt->second;

            // 只处理跨 block 的依赖
            int producerId = getSsbufferId(depDefinedOp);
            SmallVector<Operation *> depUsers = userIt->second;

            // 核心判断：如果 ALL consumers 都在同一 block，则不缓冲（normal SSA flow）
            // 如果存在跨 block 的 consumer，则需要缓冲
            if (producerId != -1) {
                bool allUsersSameBlock = true;
                for (Operation *depUser : depUsers) {
                    if (getSsbufferId(depUser) != producerId) {
                        allUsersSameBlock = false;
                        break;
                    }
                }
                if (allUsersSameBlock)
                    continue;  // 同 block，不缓冲
            }

            // Producer: 在 depDefinedOp 之后插入双缓冲写入
            // 所有新创建的 op（index_cast/divui/remsi/cmpi/scif.if/materialize_in_destination）打上 producerId 标签
            OpBuilder pb(forOp.getContext());
            pb.setInsertionPointAfter(depDefinedOp);
            SmallVector<Operation *> producerNewOps =
                insertProducerLogic(pb, depVal, buffers, forOp);
            addBlockAttrForOps(producerNewOps, producerId, globalBuilder);

            // Consumer: 只在跨 block 的 depUser 之前插入双缓冲读取
            // 同 block 内的 consumer 使用原始 depVal（normal SSA flow）
            for (Operation *depUser : depUsers) {
                bool usesDepVal = false;
                for (Value operand : depUser->getOperands()) {
                    if (operand == depVal) { usesDepVal = true; break; }
                }
                if (!usesDepVal) continue;

                // Per-consumer 过滤：只有跨 block 的 consumer 才插入缓冲读取
                int userBlockId = getSsbufferId(depUser);
                if (userBlockId == producerId) continue;  // 同 block，不缓冲

                OpBuilder cb(forOp.getContext());
                cb.setInsertionPoint(depUser);
                SmallVector<Operation *> resultIfOps;
                SmallVector<Operation *> consumerNewOps =
                    insertConsumerLogic(cb, buffers, forOp, resultIfOps);
                addBlockAttrForOps(consumerNewOps, userBlockId, globalBuilder);

                // resultIfOps 最后一个 ifOp 的第 0 个结果就是 selectedBuffer
                Operation *resultIf = resultIfOps.back();
                Value selectedBuffer = resultIf->getResult(0);

                // 记录: 这个 forOp 的 selectedBuffer 对应哪一对 buffer（只存 memref）
                SmallVector<Value> memrefs;
                for (auto &buf : buffers)
                    memrefs.push_back(buf.second);
                forOpBufferMap[forOp][selectedBuffer] = memrefs;

                // 替换 depUser 中所有对 depVal 的使用
                for (OpOperand &use : depUser->getOpOperands()) {
                    if (use.get() == depVal)
                        use.set(selectedBuffer);
                }
            }
        }
    }
}

//====================== runOnOperation ======================
struct MultiBufferPass
    : public mlir::triton::impl::MultiBufferBase<MultiBufferPass> {
    void runOnOperation() override;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect, bufferization::BufferizationDialect>();
    }
};

void MultiBufferPass::runOnOperation() {
    auto module = getOperation();
    OpBuilder builder(module.getContext());

    module.walk([&](scope::ScopeOp scope) {
        auto coreTypeAttr = scope->getAttrOfType<TCoreTypeAttr>(TCoreTypeAttr::name);
        if (!coreTypeAttr || coreTypeAttr.getTcoretype() != TCoreType::VECTOR)
            return WalkResult::advance();

        scf::ForOp mainLoopForOp;
        // Fallback 策略（优先级递减）：
        // 1. ssbuffer.mainloop 属性
        // 2. ssbuffer.block_id = 40（可能在 for op 或 body terminator）
        // 3. 有 iter_args 的最内层 scf.for
        // 优先级用 foundLevel 保证：找到更高优先级就替换，否则保留更深嵌套的
        int foundLevel = 0;
        std::function<void(Region&)> findMainloop = [&](Region &region) {
            for (Block &block : region) {
                for (Operation &op : block) {
                    if (auto f = dyn_cast<scf::ForOp>(&op)) {
                        bool hasMainloop = f->hasAttr("ssbuffer.main_loop");
                        bool bodyHasMainloop = false;
                        bool bodyHasBlockId40 = false;
                        if (auto *term = f.getBody()->getTerminator()) {
                            bodyHasMainloop = term->hasAttr("ssbuffer.main_loop");
                            if (auto a = term->getAttrOfType<IntegerAttr>("ssbuffer.block_id"))
                                bodyHasBlockId40 = (a.getInt() == 40);
                        }
                        bool opHasBlockId40 = false;
                        if (auto blockIdAttr = f->getAttrOfType<IntegerAttr>("ssbuffer.block_id"))
                            opHasBlockId40 = (blockIdAttr.getInt() == 40);
                        bool hasIterArgs = f.getNumResults() > 0 || !f.getInitArgs().empty();

                        if (hasMainloop || bodyHasMainloop) {
                            mainLoopForOp = f; foundLevel = 1; return;
                        }
                        if (opHasBlockId40 || bodyHasBlockId40) {
                            if (foundLevel < 2) { mainLoopForOp = f; foundLevel = 2; }
                        }
                        if (hasIterArgs && foundLevel < 3) {
                            mainLoopForOp = f; foundLevel = 3;
                        }
                    }
                    for (Region &r : op.getRegions())
                        findMainloop(r);
                }
            }
        };
findMainloop(scope.getBodyRegion());
        if (!mainLoopForOp)
            return WalkResult::advance();

        // Step 1: 收集 block 信息
        DenseMap<Value, BlockInfo> blocks;
        auto depValueMap = collectBlockInfo(mainLoopForOp, blocks);
        if (blocks.empty())
            return WalkResult::advance();

        // Step 2a: 构建 depUserMap
        auto depUserMap = buildDepUserMap(blocks);

        // Step 2b: 收集需要 buffer 的唯一 ShapedType（按 type 分组复用）
        auto valueList = collectBufferValues(depValueMap);
        if (valueList.empty())
            return WalkResult::advance();

        // Step 2c: 在 for 循环之前（scope 内，for 外）一次性插入所有 buffer allocation
        auto bufferMap = insertBuffersBeforeFor(mainLoopForOp, valueList, builder);

        // Step 3: 插入 producer/consumer 双缓冲控制流
        // forOpBufferMap: 存入全局单例，供下游 AddIfControls Pass 使用
        ForOpSelectedBufferMap forOpBufferMap;
        addMultiBuffCaculate(mainLoopForOp, depValueMap, bufferMap, blocks, depUserMap,
                             forOpBufferMap);

        // Step 4: 将 forOpBufferMap 写入全局单例，供下游 Pass 直接读取
        //
        // InnerMultibuffer 在 IR 中插入了 scf.if + to_tensor 的 ping-pong 模式：
        //   %selectedBuffer = scf.if %cond -> (tensor<X>) {
        //     then:   %t = bufferization.to_tensor %memref0 : memref<X>
        //     else:   %e = bufferization.to_tensor %memref1 : memref<X>
        //   }
        //
        // 下游 AddIfControlsPass 直接从全局单例读取：
        //
        //   auto &rel = mlir::triton::getGlobalBufferRelation();
        //   auto *bufferMap = rel.getBufferRelations(mainLoopForOp);
        //   for (auto &[selBuf, memrefs] : *bufferMap) {
        //     // selBuf: scf.if 结果（ping-pong selector）
        //     // memrefs[0], memrefs[1]: 两个 ping-pong memref
        //   }
        //
        auto &globalRel = mlir::triton::getGlobalBufferRelation();
        globalRel.clear();
        for (auto &entry : forOpBufferMap) {
          scf::ForOp forOp = entry.first;
          for (auto &innerEntry : entry.second) {
            Value selectedBuffer = innerEntry.first;
            SmallVector<Value> memrefs = innerEntry.second;
            globalRel.addBufferRelation(forOp, selectedBuffer, memrefs);
          }
        }

        // ================================================================
        // DEBUG: 打印写入全局单例的数据（供下游 AddIfControls 读取）
        // DEBUG_MODE 控制输出模式（见文件顶部）
        // ================================================================
#if DEBUG_MODE >= 1
        llvm::errs() << "\n============================================================\n";
        llvm::errs() << "[InnerMultibuffer] >>> 写入全局单例 BufferRelationAnalysis\n";
        llvm::errs() << "============================================================\n";

        const auto &allRels = globalRel.getAllBufferRelations();
        llvm::errs() << "forOp 数量: " << allRels.size() << "\n";

        for (auto &entry : allRels) {
          scf::ForOp forOp = entry.first;
          const auto &innerMap = entry.second;

          llvm::errs() << "\n[forOp @" << forOp << "]\n";
          llvm::errs() << "  selectedBuffer 数量: " << innerMap.size() << "\n";

          for (auto &innerEntry : innerMap) {
            Value selectedBuffer = innerEntry.first;
            const SmallVector<Value> &memrefs = innerEntry.second;

            llvm::errs() << "  selectedBuffer: " << selectedBuffer << "\n";
            llvm::errs() << "    SSA name: ";
            if (auto *op = selectedBuffer.getDefiningOp())
              llvm::errs() << "%" << op->getName().getStringRef() << "\n";
            else
              llvm::errs() << "(block arg)\n";
            llvm::errs() << "    Type: " << selectedBuffer.getType() << "\n";
            llvm::errs() << "    memrefs 数量: " << memrefs.size() << "\n";

            for (auto [i, v] : llvm::enumerate(memrefs)) {
              llvm::errs() << "      memref[" << i << "]: " << v << "\n";
              llvm::errs() << "        SSA name: ";
              if (auto *op = v.getDefiningOp())
                llvm::errs() << "%" << op->getName().getStringRef() << "\n";
              else
                llvm::errs() << "(block arg)\n";
              llvm::errs() << "        Type: " << v.getType() << "\n";
            }
          }
        }

        const auto &crossCore = globalRel.getCrossCoreRelations();
        llvm::errs() << "\ncrossCoreRelations 大小: " << crossCore.size() << "\n";
        for (auto &entry : crossCore) {
          llvm::errs() << "  consumer: " << entry.first << "\n";
          llvm::errs() << "    producers: ";
          for (auto v : entry.second)
            llvm::errs() << v << " ";
          llvm::errs() << "\n";
        }

        llvm::errs() << "============================================================\n";
        llvm::errs() << "[InnerMultibuffer] <<< 写入完成\n";
        llvm::errs() << "============================================================\n";
        llvm::errs().flush();
#endif // DEBUG_MODE >= 1

#if DEBUG_MODE == 1
        // DEBUG_MODE==1 时静默跳过 IR 打印（return 空 IR）
        return WalkResult::advance();
#endif

        return WalkResult::advance();
    });
}

} // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createMultiBufferPass() {
    return std::make_unique<MultiBufferPass>();
}
