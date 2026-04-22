#include "TritonAffinityOpt/Passes.h"

#include "llvm/ADT/APFloat.h"
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>

#include "TritonAffinityOpt/BufferRelationAnalysis.h"
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ADDIFCONTROLS
#include "ascend/include/TritonAffinityOpt/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hivm;
using namespace triton;

namespace {
  // 变量更新类型：记录控制变量在 IfOp then 块中需要执行的操作
  enum class VarUpdateType {
    INC,    // +1 (生产者变量)
    DEC     // -1 (消费者变量)
  };

  struct OutputGroupInfo {
    SmallVector<Value> outputs;      // 输出列表
    SmallVector<Value> inputVars;    // 输入变量列表
  };
  struct AddIfControlsPass
      : public mlir::triton::impl::AddIfControlsBase<
            AddIfControlsPass> {
    DenseMap<scf::ForOp, SmallVector<int>> blockCounters; // 计算块独立的计数器的index
    DenseMap<scf::ForOp, int> blockCounterNums; // 计算块的计数器的数量
    DenseMap<scf::ForOp, SmallVector<int>> innerDepConds; // 内部依赖的控制条件的index
    //TODO:获取依赖map
    DenseMap<Value, SmallVector<Value>> crossCoreDependentMap;
    DenseMap<scf::ForOp, DenseMap<Value, SmallVector<Value>>> intraCoreDependentMap;
    DenseMap<scf::IfOp, Value> cntArgs; // IfOp使用的计算器参数
    SmallVector<Value> currentUsedVars; // 当前正在处理的 IfOp 使用的控制变量

    // 全局映射：记录控制变量的最新值（用于多个 IfOp 链式更新）
    // key: 原始控制变量（for 循环的迭代参数）
    // value: 该变量当前的最新值（可能是某个 IfOp 的结果）
    DenseMap<Value, Value> controlVarToLatestValue;



    void runOnOperation() override;
    void getDependentDialects(DialectRegistry &registry) const override {
      registry.insert<LLVM::LLVMDialect>();
      registry.insert<linalg::LinalgDialect>();
    }
    void createIfOps(ModuleOp module);
    void updateForOps(ModuleOp module);
    void updateIfConds(ModuleOp module);
    void initTestData(ModuleOp module);
    SmallVector<SmallVector<Value>> initSSBuffer(ModuleOp module);
    void UpdateForIterTimes(ModuleOp module);
    void initDependentMap(ModuleOp module);
    void updateInputAndInitMap(ModuleOp module);

    void collectOpsByBlockId(scf::ForOp forOp, DenseMap<int, SmallVector<Operation *>> &blockOps);
    void replaceExternalIfOpUses(scf::IfOp ifOp, ArrayRef<Value> oldYieldValues);
    void computeYieldValues(scf::ForOp forOp, DenseMap<int, SmallVector<Operation *>> &blockOps,
                        DenseMap<int, SmallVector<Value>> &thenYieldValues,
                        DenseMap<int, SmallVector<Value>> &elseYieldValues);
    void createIfInMainloop(scf::ForOp forOp, DenseMap<int, SmallVector<Operation *>> &blockOps,
                     DenseMap<int, SmallVector<Value>> &thenYieldValues,
                     DenseMap<int, SmallVector<Value>> &elseYieldValues);
    void createIfOpsByBlockId(scf::ForOp forOp);
    void copyOpsInCube(ModuleOp module);
    void addBlockCounters(ModuleOp module);
    void addInnerDepConds(ModuleOp module);
    void insertInterCorePipeSForCube(ModuleOp module);
    void insertInterCorePipeSForVector(ModuleOp module);
    void insertInterCorePipeS(ModuleOp module);

    Value getVarValue(scf::ForOp forOp, int varIndex);

    void collectDependencyBuffers(scf::ForOp forOp,
                                  DenseMap<int, DenseMap<Value, SmallVector<Value>>> &crossCoreBuffers,
                                  DenseMap<int, DenseMap<Value, SmallVector<Value>>> &intraCoreBuffers);

// 扩展crossCoreBuffers，添加通过annotation.mark标记的等价value到SmallVector中
// 返回: 新的DenseMap<int, DenseMap<Value, SmallVector<Value>>>，包含等价value
DenseMap<int, DenseMap<Value, SmallVector<Value>>> extendCrossCoreBuffersWithEquivalentValues(
                              ModuleOp module,
                              DenseMap<int, DenseMap<Value, SmallVector<Value>>> crossCoreBuffers);

    void getInputOutputValues(scf::IfOp ifOp,
                              DenseMap<int, DenseMap<Value, SmallVector<Value>>> crossCoreBuffers,
                              DenseMap<int, DenseMap<Value, SmallVector<Value>>> intraCoreBuffers,
                              SmallVector<int> &crossCoreInputValues,
                              SmallVector<int> &crossCoreOutputValues,
                              SmallVector<int> &intraCoreInputValues,
                              SmallVector<int> &intraCoreOutputValues);

    // 构建生产者组信息
    SmallVector<OutputGroupInfo> buildOutputGroups(SmallVector<int> &intraCoreOutputValues,
                                                   DenseMap<int, DenseMap<Value, SmallVector<Value> > > &
                                                   intraCoreBuffers,
                                                   DenseMap<int, Value> &idxToVar);

    // 创建新 IfOp 并处理 then/else 块
    scf::IfOp createNewIfOpWithBlocks(scf::IfOp oldIfOp, Value combinedCond,
                                      DenseMap<Value, VarUpdateType> &varUpdateTypes,
                                      bool hasCounter, Value counter,
                                      Value step);




    Value setIntraCoreCondition(ModuleOp module, scf::IfOp ifOp,
                                DenseMap<int, DenseMap<Value, SmallVector<Value> > > &intraCoreBuffers,
                                SmallVector<int> &intraCoreInputIndices, SmallVector<int> &intraCoreOutputIndices,
                                DenseMap<int, Value> &idxToVar,
                                DenseMap<Value, VarUpdateType> &varUpdateTypes);

    // 更新 for 循环的 yield
    void updateForOpYield(scf::ForOp forOp, scf::IfOp newIfOp, scf::IfOp oldIfOp, bool hasCounter, Value counter);

    void combineConditions(ModuleOp module, Value crossCoreCond, Value intraCoreCond,
                           scf::IfOp ifOp, scf::ForOp forOp,
                           size_t &usedCounterNum,
                           DenseMap<Value, VarUpdateType> &varUpdateTypes);

    Value setCrossCoreCondition(SmallVector<int> crossCoreInputValues,
                                 SmallVector<int> crossCoreOutputValues,
                                 DenseMap<int, DenseMap<Value, SmallVector<Value>>> &crossCoreBuffers,
                                 scf::IfOp ifOp,
                                 SmallVector<SmallVector<Value>> ssbufferPtrs);
  };
} // namespace

void AddIfControlsPass::collectOpsByBlockId(
    scf::ForOp forOp,
    DenseMap<int, SmallVector<Operation *>> &blockOps) {

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto attr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
      int id = attr.getInt();
      blockOps[id].push_back(&op);
    }
  }
}

void collectAllNestedOps(Operation *op, DenseSet<Operation *> &regionOps) {
  if (!op || regionOps.contains(op))
    return;

  regionOps.insert(op);
  for (Region &region : op->getRegions()) {
    for (Block &block : region)
      for (Operation &nestedOp : block)
        collectAllNestedOps(&nestedOp, regionOps);
  }
}

Value findIterArgInMainLoop(Value v, Type t) {
  for (Operation *user : v.getUsers()) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
        for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
          if (operand.getAsOpaquePointer() == v.getAsOpaquePointer()) {
            Value iterArg = forOp.getRegionIterArgs()[idx];
            if (iterArg.getType() == t)
              return iterArg;
          }
        }
      }
    }
  }

  std::string ifOpStr;
  llvm::raw_string_ostream(ifOpStr) << *v.getDefiningOp();
  std::string errorMsg = "ifOp: " + ifOpStr + " else yield value not found in forOp iter_args.";
  llvm_unreachable(errorMsg.c_str());
  return v;
}

void AddIfControlsPass::replaceExternalIfOpUses(scf::IfOp ifOp, ArrayRef<Value> oldYieldValues) {
  for (size_t i = 0; i < oldYieldValues.size(); ++i) {
    Value oldVal = oldYieldValues[i];
    Value newVal = ifOp.getResult(i);
    SmallVector<OpOperand *> usesToReplace;

    for (OpOperand &use : llvm::make_early_inc_range(oldVal.getUses())) {
      Operation *user = use.getOwner();
      if (ifOp->isAncestor(user))
        continue;
      if (user->getBlock() == ifOp->getBlock()) {
        if (!ifOp->isBeforeInBlock(user))
          continue;
      }
      usesToReplace.push_back(&use);
    }

    for (OpOperand *use : usesToReplace)
      use->set(newVal);
  }
}

void AddIfControlsPass::computeYieldValues(scf::ForOp forOp, DenseMap<int, SmallVector<Operation *>> &blockOps,
                        DenseMap<int, SmallVector<Value>> &thenYieldValues,
                        DenseMap<int, SmallVector<Value>> &elseYieldValues) {
  for (auto &p : blockOps) {
    int id = p.first;
    SmallVector<Operation *> &ops = p.second;

    DenseSet<Operation *> regionOps;
    for (Operation *op : ops)
      collectAllNestedOps(op, regionOps);

    SmallVector<Value> yieldValues;
    for (Operation *op : ops) {
      for (Value res : op->getResults()) {
        bool needsYield = false;
        for (OpOperand &use : res.getUses()) {
          Operation *user = use.getOwner();
          if (!regionOps.contains(user)) {
            needsYield = true;
            break;
          }
        }
        if (needsYield) {
          yieldValues.push_back(res);
        }
      }
    }

    thenYieldValues[id] = yieldValues;
    elseYieldValues[id].clear();
    elseYieldValues[id].reserve(yieldValues.size());
    for (Value v : yieldValues)
      elseYieldValues[id].push_back(findIterArgInMainLoop(v, v.getType()));
  }
}

void AddIfControlsPass::createIfInMainloop(scf::ForOp forOp, DenseMap<int, SmallVector<Operation *>> &blockOps,
                     DenseMap<int, SmallVector<Value>> &thenYieldValues,
                     DenseMap<int, SmallVector<Value>> &elseYieldValues) {
  SmallVector<int> ids;
  for (auto &p : blockOps)
    ids.push_back(p.first);
  llvm::sort(ids);

  for (int id : ids) {
    SmallVector<Operation *> &ops = blockOps[id];
    if (ops.empty())
      continue;

    Operation *insertPoint = ops.front();
    OpBuilder builder(insertPoint);
    Location loc = insertPoint->getLoc();

    SmallVector<Value> &thenValues = thenYieldValues[id];
    SmallVector<Value> &elseValues = elseYieldValues[id];
    bool needsYield = !thenValues.empty();

    SmallVector<Type> resultTypes;
    for (Value v : thenValues)
      resultTypes.push_back(v.getType());

    scf::IfOp ifOp;
    if (needsYield)
      ifOp = builder.create<scf::IfOp>(loc, resultTypes, builder.create<arith::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true)), true);
    else
      ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, builder.create<arith::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true)), false);

    ifOp->setAttr("ssbuffer.if", builder.getI32IntegerAttr(id));

    Block &thenBlock = ifOp.getThenRegion().front();
    for (Operation *op : llvm::reverse(ops))
      op->moveBefore(&thenBlock, thenBlock.begin());

    if (needsYield) {
      OpBuilder thenBuilder(&thenBlock, thenBlock.end());
      thenBuilder.create<scf::YieldOp>(loc, thenValues);

      Block &elseBlock = ifOp.getElseRegion().front();
      OpBuilder elseBuilder(&elseBlock, elseBlock.end());
      elseBuilder.create<scf::YieldOp>(loc, elseValues);

      replaceExternalIfOpUses(ifOp, thenValues);
    }
  }
}

void AddIfControlsPass::createIfOpsByBlockId(scf::ForOp forOp) {
  DenseMap<int, SmallVector<Operation *>> blockOps;
  collectOpsByBlockId(forOp, blockOps);

  blockCounterNums[forOp] = blockOps.size();

  DenseMap<int, SmallVector<Value>> thenYieldValues;
  DenseMap<int, SmallVector<Value>> elseYieldValues;
  computeYieldValues(forOp, blockOps, thenYieldValues, elseYieldValues);

  createIfInMainloop(forOp, blockOps, thenYieldValues, elseYieldValues);
}

void collectDependencyClosure(Operation *op,
                              DenseSet<Operation *> &closure,
                              DenseSet<Operation *> &scopeOps) {
  if (!op || closure.contains(op))
    return;

  if (!scopeOps.contains(op))
    return;

  closure.insert(op);

  for (Value operand : op->getOperands()) {
    if (Operation *def = operand.getDefiningOp()) {
      collectDependencyClosure(def, closure, scopeOps);
    }
  }
}

void stableTopoSort(DenseSet<Operation *> &ops,
                    DenseMap<Operation*, int> &opOrder,
                    SmallVectorImpl<Operation *> &sorted) {

  DenseSet<Operation *> visited;

  std::function<void(Operation *)> dfs = [&](Operation *op) {
    if (!op || visited.contains(op))
      return;

    visited.insert(op);
    SmallVector<Operation*> deps;
    for (Value operand : op->getOperands()) {
      if (Operation *def = operand.getDefiningOp()) {
        if (ops.contains(def))
          deps.push_back(def);
      }
    }

    llvm::sort(deps, [&](Operation *a, Operation *b) {
      return opOrder[a] < opOrder[b];
    });

    for (Operation *dep : deps)
      dfs(dep);

    sorted.push_back(op);
  };

  SmallVector<Operation*> opList(ops.begin(), ops.end());
  llvm::sort(opList, [&](Operation *a, Operation *b) {
    return opOrder[a] < opOrder[b];
  });

  for (Operation *op : opList)
    dfs(op);
}

void AddIfControlsPass::copyOpsInCube(ModuleOp module) {
  module.walk([&](scope::ScopeOp scopeOp) {

    auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
    if (!attr || attr != hivm::TCoreTypeAttr::get(module.getContext(),
                                                  hivm::TCoreType::CUBE))
      return;

    scopeOp.walk([&](scf::ForOp forOp) {

      if (!forOp->hasAttr("ssbuffer.main_loop"))
        return;

      DenseSet<Operation *> scopeOps;
      for (Operation &op : forOp.getBody()->without_terminator())
        collectAllNestedOps(&op, scopeOps);

      DenseMap<int, SmallVector<Operation *>> blockOps;
      collectOpsByBlockId(forOp, blockOps);

      SmallVector<int> ids;
      for (auto &p : blockOps)
        ids.push_back(p.first);

      llvm::sort(ids, std::greater<int>());

      DenseMap<Operation*, int> opOrder;
      int idx = 0;
      for (Operation &op : *forOp.getBody()) {
        opOrder[&op] = idx++;
      }

      for (int id : ids) {
        auto &ops = blockOps[id];
        if (ops.empty())
          continue;

        DenseSet<Operation *> groupOps;
        for (Operation *op : ops)
          collectAllNestedOps(op, groupOps);

        DenseSet<Operation *> depClosure;

        for (Operation *op : ops) {
          for (Value operand : op->getOperands()) {
            if (Operation *def = operand.getDefiningOp()) {
              if (scopeOps.contains(def) && !groupOps.contains(def)) {
                collectDependencyClosure(def, depClosure, scopeOps);
              }
            }
          }
        }

        // Also trace forward through uses - for memref.copy whose second operand is from
        // memref.alloc, include the copy op and trace its dependencies
        bool changed = true;
        SmallVector<Operation *> toProcess(depClosure.begin(), depClosure.end());
        while (changed) {
          changed = false;
          SmallVector<Operation *> newlyAdded;
          for (Operation *op : toProcess) {
            for (OpOperand &use : op->getUses()) {
              Operation *user = use.getOwner();
              if (scopeOps.contains(user) && !groupOps.contains(user) &&
                  !depClosure.contains(user)) {
                // Only handle memref.copy when its second operand is from memref.alloc
                if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
                  Value secondOperand = copyOp.getOperand(1);
                  if (Operation *def = secondOperand.getDefiningOp()) {
                    if (isa<memref::AllocOp>(def)) {
                      depClosure.insert(user);
                      newlyAdded.push_back(user);
                      changed = true;
                      // Trace backward from this user to get its dependencies
                      for (Value operand : user->getOperands()) {
                        if (Operation *defOp = operand.getDefiningOp()) {
                          if (scopeOps.contains(defOp) && !groupOps.contains(defOp)) {
                            collectDependencyClosure(defOp, depClosure, scopeOps);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          toProcess = std::move(newlyAdded);
        }

        if (depClosure.empty())
          continue;

        SmallVector<Operation*> sortedDeps;
        stableTopoSort(depClosure, opOrder, sortedDeps);

        IRMapping mapper;
        OpBuilder builder(ops.front());

        for (Operation *depOp : sortedDeps) {
          Operation *cloned = depOp->clone(mapper);
          builder.insert(cloned);

          for (auto &attr : depOp->getAttrs()) {
            cloned->setAttr(attr.getName(), attr.getValue());
          }
          cloned->setAttr("ssbuffer.block_id",
                          builder.getI32IntegerAttr(id));

          mapper.map(depOp, cloned);
        }

        for (Operation *op : ops) {
          for (OpOperand &operand : op->getOpOperands()) {
            Value oldVal = operand.get();
            Value newVal = mapper.lookupOrDefault(oldVal);
            if (newVal != oldVal)
              operand.set(newVal);
          }
        }
      }
    });
  });
}

void AddIfControlsPass::createIfOps(ModuleOp module) {

  copyOpsInCube(module);

  module.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr("ssbuffer.main_loop")) {
      createIfOpsByBlockId(forOp);
    }
  });
}

void AddIfControlsPass::addBlockCounters(ModuleOp module) {

  SmallVector<scf::ForOp> forOpsToProcess;
  module.walk([&](scf::ForOp forOp) {
    if (blockCounterNums.count(forOp))
      forOpsToProcess.push_back(forOp);
  });

  for (scf::ForOp oldForOp : forOpsToProcess) {
    int numCounters = blockCounterNums[oldForOp];
    if (numCounters == 0)
      continue;

    OpBuilder builder(oldForOp);
    Location loc = oldForOp.getLoc();

    SmallVector<Value> newInitArgs(oldForOp.getInitArgs().begin(),
                                  oldForOp.getInitArgs().end());

    for (int i = 0; i < numCounters; ++i) {
      Value zero = builder.create<arith::ConstantOp>(
          loc, builder.getI32Type(),
          builder.getI32IntegerAttr(0));
      newInitArgs.push_back(zero);
    }

    scf::ForOp newForOp = builder.create<scf::ForOp>(
        loc,
        oldForOp.getLowerBound(),
        oldForOp.getUpperBound(),
        oldForOp.getStep(),
        newInitArgs);

    for (auto &attr : oldForOp->getAttrs()) {
      newForOp->setAttr(attr.getName(), attr.getValue());
    }

    Block *oldBlock = oldForOp.getBody();
    Block *newBlock = newForOp.getBody();

    unsigned oldNum = oldForOp.getNumRegionIterArgs();
    unsigned totalArgs = oldBlock->getNumArguments();
    for (unsigned i = 0; i < totalArgs; ++i) {
      BlockArgument oldArg = oldBlock->getArgument(i);
      BlockArgument newArg = newBlock->getArgument(i);

      oldArg.replaceAllUsesWith(newArg);
    }

    Operation *oldTerminator = oldBlock->getTerminator();

    for (Operation &op :
         llvm::make_early_inc_range(oldBlock->without_terminator())) {
      op.moveBefore(newBlock, newBlock->end());
    }

    auto oldYield = cast<scf::YieldOp>(oldTerminator);

    SmallVector<Value> newYieldOperands;

    for (unsigned i = 0; i < oldNum; ++i) {
      newYieldOperands.push_back(oldYield.getOperand(i));
    }

    for (int i = 0; i < numCounters; ++i) {
      Value counterArg = newBlock->getArgument(1 + oldNum + i);
      newYieldOperands.push_back(counterArg);
    }

    builder.setInsertionPointToEnd(newBlock);
    builder.create<scf::YieldOp>(loc, newYieldOperands);

    oldYield.erase();

    SmallVector<int> counterIndices;
    for (int i = 0; i < numCounters; ++i) {
        counterIndices.push_back(oldNum + i);
    }
    blockCounters[newForOp] = counterIndices;

    if (oldForOp.getNumResults() > 0) {
      SmallVector<Value> newResults;
      for (unsigned i = 0; i < oldForOp.getNumResults(); ++i) {
        newResults.push_back(newForOp.getResult(i));
      }
      oldForOp.replaceAllUsesWith(newResults);
    }

    // Update intraCoreDependentMap: remap from oldForOp to newForOp
    if (intraCoreDependentMap.count(oldForOp)) {
      intraCoreDependentMap[newForOp] = intraCoreDependentMap[oldForOp];
      intraCoreDependentMap.erase(oldForOp);
    }
    oldForOp.erase();
  }
}

void AddIfControlsPass::addInnerDepConds(ModuleOp module) {
  // // 构造输入：假设每个ssbuffer.main_loop的forop都只有1个innerdeps
  // DenseMap<scf::ForOp, int> InnerDepNums;
  // module.walk([&](scf::ForOp forOp) {
  //   if (forOp->hasAttr("ssbuffer.main_loop")) {
  //     InnerDepNums[forOp] = 1;
  //   }
  // });

  DenseMap<scf::ForOp, int> InnerDepNums;
  for (auto& entry : intraCoreDependentMap) {
    scf::ForOp forOp = entry.first;
    auto& innerMap = entry.second;
    int count = innerMap.size();
    InnerDepNums[forOp] = count;
  }

  SmallVector<scf::ForOp> forOpsToProcess;
  module.walk([&](scf::ForOp forOp) {
    if (InnerDepNums.count(forOp))
      forOpsToProcess.push_back(forOp);
  });

  for (scf::ForOp oldForOp : forOpsToProcess) {
    int numInnerDeps = InnerDepNums[oldForOp];
    if (numInnerDeps == 0)
      continue;

    OpBuilder builder(oldForOp);
    Location loc = oldForOp.getLoc();

    SmallVector<Value> newInitArgs(oldForOp.getInitArgs().begin(),
                                  oldForOp.getInitArgs().end());
    for (int i = 0; i < numInnerDeps; ++i) {
      Value zero = builder.create<arith::ConstantOp>(
          loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
      newInitArgs.push_back(zero);
    }

    scf::ForOp newForOp = builder.create<scf::ForOp>(
        loc,
        oldForOp.getLowerBound(),
        oldForOp.getUpperBound(),
        oldForOp.getStep(),
        newInitArgs);

    for (auto &attr : oldForOp->getAttrs())
      newForOp->setAttr(attr.getName(), attr.getValue());

    Block *oldBlock = oldForOp.getBody();
    Block *newBlock = newForOp.getBody();
    unsigned oldNum = oldForOp.getNumRegionIterArgs();
    unsigned totalArgs = oldBlock->getNumArguments();
    for (unsigned i = 0; i < totalArgs; ++i) {
      BlockArgument oldArg = oldBlock->getArgument(i);
      BlockArgument newArg = newBlock->getArgument(i);
      oldArg.replaceAllUsesWith(newArg);
    }

    Operation *oldTerminator = oldBlock->getTerminator();
    for (Operation &op : llvm::make_early_inc_range(oldBlock->without_terminator())) {
      op.moveBefore(newBlock, newBlock->end());
    }

    auto oldYield = cast<scf::YieldOp>(oldTerminator);
    SmallVector<Value> newYieldOperands;
    for (unsigned i = 0; i < oldNum; ++i)
      newYieldOperands.push_back(oldYield.getOperand(i));

    for (int i = 0; i < numInnerDeps; ++i) {
      newYieldOperands.push_back(newBlock->getArgument(1 + oldNum + i));
    }

    builder.setInsertionPointToEnd(newBlock);
    builder.create<scf::YieldOp>(loc, newYieldOperands);
    oldYield.erase();

    SmallVector<int> oldBlockCounterIndices;
    if (blockCounters.count(oldForOp)) {
      oldBlockCounterIndices = blockCounters[oldForOp];
      blockCounters.erase(oldForOp);
    }
    blockCounters[newForOp] = oldBlockCounterIndices;

    SmallVector<int> innerDepIndices;
    for (int i = 0; i < numInnerDeps; ++i)
      innerDepIndices.push_back(oldNum + i);
    innerDepConds[newForOp] = innerDepIndices;

    if (oldForOp.getNumResults() > 0) {
      SmallVector<Value> newResults;
      for (unsigned i = 0; i < oldForOp.getNumResults(); ++i)
        newResults.push_back(newForOp.getResult(i));
      oldForOp.replaceAllUsesWith(newResults);
    }

    // Update intraCoreDependentMap: remap from oldForOp to newForOp
    if (intraCoreDependentMap.count(oldForOp)) {
      intraCoreDependentMap[newForOp] = intraCoreDependentMap[oldForOp];
      intraCoreDependentMap.erase(oldForOp);
    }
    oldForOp.erase();
  }
}

void AddIfControlsPass::insertInterCorePipeSForCube(ModuleOp module) {
  auto cubeCoreType = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
  auto setPipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
  auto waitPipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
  int cubeFlagId = 15;
  int vectorFlagId = 14;

  module.walk([&](scope::ScopeOp scopeOp) {
    auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
    if (!attr || attr != cubeCoreType)
      return;

    Block &scopeBlock = scopeOp.getRegion().front();
    Operation *scopeTerminator = scopeBlock.getTerminator();
    OpBuilder scopeBuilder(scopeTerminator);
    Location scopeLoc = scopeTerminator->getLoc();
    auto scopeFlagId = scopeBuilder.getIntegerAttr(scopeBuilder.getI64Type(), cubeFlagId);
    scopeBuilder.setInsertionPoint(scopeTerminator);
    scopeBuilder.create<SyncBlockWaitOp>(scopeLoc, cubeCoreType, setPipeType, waitPipeType, scopeFlagId);

    scopeOp.walk([&](scf::ForOp forOp) {
      if (!forOp->hasAttr("ssbuffer.main_loop"))
        return;

      Block &forBody = forOp.getRegion().front();
      OpBuilder forStartBuilder(&forBody, forBody.begin());
      Location forStartLoc = forOp.getLoc();
      auto forWaitFlagId = scopeBuilder.getIntegerAttr(scopeBuilder.getI64Type(), cubeFlagId);
      forStartBuilder.create<SyncBlockWaitOp>(forStartLoc, cubeCoreType, setPipeType, waitPipeType, forWaitFlagId);

      Operation *forTerminator = forBody.getTerminator();
      OpBuilder forYieldBuilder(forTerminator);
      Location forYieldLoc = forTerminator->getLoc();
      auto forSetFlagId = scopeBuilder.getIntegerAttr(scopeBuilder.getI64Type(), vectorFlagId);
      forYieldBuilder.setInsertionPoint(forTerminator);
      forYieldBuilder.create<SyncBlockSetOp>(forYieldLoc, cubeCoreType, setPipeType, waitPipeType, forSetFlagId);
    });
  });
}

void AddIfControlsPass::insertInterCorePipeSForVector(ModuleOp module) {
  auto vectorCoreType = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
  auto setPipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
  auto waitPipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
  int cubeFlagId = 15;
  int vectorFlagId = 14;

  module.walk([&](scope::ScopeOp scopeOp) {
    auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
    if (!attr || attr != vectorCoreType)
      return;

    Block &scopeBlock = scopeOp.getRegion().front();
    OpBuilder scopeStartBuilder(&scopeBlock, scopeBlock.begin());
    Location scopeLoc = scopeOp.getLoc();
    auto scopeSetFlagId = scopeStartBuilder.getIntegerAttr(scopeStartBuilder.getI64Type(), cubeFlagId);
    scopeStartBuilder.create<SyncBlockSetOp>(scopeLoc, vectorCoreType, setPipeType, waitPipeType, scopeSetFlagId);

    scopeOp.walk([&](scf::ForOp forOp) {
      if (!forOp->hasAttr("ssbuffer.main_loop"))
        return;

      Block &forBody = forOp.getRegion().front();
      OpBuilder forStartBuilder(&forBody, forBody.begin());
      Location forStartLoc = forOp.getLoc();
      auto forWaitFlagId = scopeStartBuilder.getIntegerAttr(scopeStartBuilder.getI64Type(), vectorFlagId);
      forStartBuilder.create<SyncBlockWaitOp>(forStartLoc, vectorCoreType, setPipeType, waitPipeType, forWaitFlagId);

      Operation *forTerminator = forBody.getTerminator();
      OpBuilder forYieldBuilder(forTerminator);
      Location forYieldLoc = forTerminator->getLoc();
      auto forSetFlagId = scopeStartBuilder.getIntegerAttr(scopeStartBuilder.getI64Type(), cubeFlagId);
      forYieldBuilder.setInsertionPoint(forTerminator);
      forYieldBuilder.create<SyncBlockSetOp>(forYieldLoc, vectorCoreType, setPipeType, waitPipeType, forSetFlagId);
    });
  });
}

void AddIfControlsPass::insertInterCorePipeS(ModuleOp module) {
  insertInterCorePipeSForCube(module);
  insertInterCorePipeSForVector(module);
}

void AddIfControlsPass::updateForOps(ModuleOp module) {
  addBlockCounters(module);

  addInnerDepConds(module);

  insertInterCorePipeS(module);
}

SmallVector<SmallVector<Value>> AddIfControlsPass::initSSBuffer(ModuleOp module)
{
  OpBuilder builder(module.getContext());
  auto i64Type = builder.getIntegerType(64);
  auto i32Type = builder.getIntegerType(32);
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext(), 11);

  int numBuffers = crossCoreDependentMap.size();
  SmallVector<SmallVector<Value>> ssbufferPtrs(2);

  module->walk([&](Operation* op) {
    if (auto scopeOp = dyn_cast<scope::ScopeOp>(op)) {
      builder.setInsertionPoint(scopeOp);
      auto zeroConst = builder.create<mlir::LLVM::ConstantOp>(scopeOp->getLoc(), i32Type, builder.getIntegerAttr(i32Type, 0));

      for (int i = 0; i < numBuffers; i++) {
        auto addr0Attr = builder.getIntegerAttr(i64Type, i * 4);
        auto addr1Attr = builder.getIntegerAttr(i64Type, 1024 + i * 4);

        auto addr0Const = builder.create<mlir::LLVM::ConstantOp>(scopeOp->getLoc(), i64Type, addr0Attr);
        auto addr1Const = builder.create<mlir::LLVM::ConstantOp>(scopeOp->getLoc(), i64Type, addr1Attr);

        auto ptr0 = builder.create<mlir::LLVM::IntToPtrOp>(scopeOp->getLoc(), ptrType, addr0Const.getResult());
        auto ptr1 = builder.create<mlir::LLVM::IntToPtrOp>(scopeOp->getLoc(), ptrType, addr1Const.getResult());

        builder.create<LLVM::StoreOp>(scopeOp->getLoc(), zeroConst, ptr0);
        builder.create<LLVM::StoreOp>(scopeOp->getLoc(), zeroConst, ptr1);

        ssbufferPtrs[0].push_back(ptr0.getResult());
        ssbufferPtrs[1].push_back(ptr1.getResult());
      }
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  return ssbufferPtrs;
}

// setCrossCoreCondition: 为if计算块设置核间buffer的控制条件
// crossCoreInputValues: 输入buffer对应的组别索引
// crossCoreOutputValues: 输出buffer对应的组别索引
// crossCoreBuffers: 每组buffer的列表，用于获取每组buffer的数量 {groupIdx -> {input: [outputs]}}
// ifOp: 需要设置条件的if操作
// ssbufferPtrs: ssbuffer的地址指针，[0]为vec0的ptr列表，[1]为vec1的ptr列表
Value AddIfControlsPass::setCrossCoreCondition(SmallVector<int> crossCoreInputValues,
                            SmallVector<int> crossCoreOutputValues,
                            DenseMap<int, DenseMap<Value, SmallVector<Value>>> &crossCoreBuffers,
                            scf::IfOp ifOp,
                            SmallVector<SmallVector<Value>> ssbufferPtrs)
{
  OpBuilder builder(ifOp);
  Location loc = ifOp.getLoc();

  // 判断当前scope是Cube还是Vector
  auto aiCAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);
  bool isAIC = false;
  mlir::Operation* parentOp = ifOp->getParentOp();
  mlir::Operation* scopeOp = nullptr;
  while (parentOp) {
    if (dyn_cast<scope::ScopeOp>(parentOp)) {
      scopeOp = parentOp;
      break;
    }
    parentOp = parentOp->getParentOp();
  }
  if (scopeOp && scopeOp->hasAttr("hivm.tcore_type")) {
    auto attr = scopeOp->getAttr("hivm.tcore_type");
    if (attr == aiCAttr) {
      isAIC = true;
    }
  }

  Value zeroConst = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value oneConst = builder.create<arith::ConstantIntOp>(loc, 1, 32);

  Value conditions = nullptr;
  Value ssbAddrOffset = nullptr;

  // 收集所有涉及的buffer组索引
  SmallVector<int> allGroupIndices;
  DenseSet<int> uniqueIndices;
  for (int idx : crossCoreInputValues) {
    if (uniqueIndices.insert(idx).second) {
      allGroupIndices.push_back(idx);
    }
  }
  for (int idx : crossCoreOutputValues) {
    if (uniqueIndices.insert(idx).second) {
      allGroupIndices.push_back(idx);
    }
  }

  // Vector模式: 预计算所有buffer组的指针，避免重复计算
  // Cube模式: 直接使用ssbufferPtrs中的指针
  DenseMap<int, Value> precomputedPtrs;
  if (!isAIC) {
    builder.setInsertionPointToStart(&scopeOp->getRegion(0).front());
    int vec1Offset = 1024;
    Value vec1OffsetValue = builder.create<arith::ConstantIntOp>(loc, vec1Offset, 64);
    auto subIdOp = builder.create<GetSubBlockIdxOp>(loc, builder.getIntegerType(64));
    ssbAddrOffset = builder.create<arith::MulIOp>(loc, subIdOp, vec1OffsetValue);
    for (int groupIdx : allGroupIndices) {
      auto ssbBaseAddr = builder.create<arith::ConstantIntOp>(loc, groupIdx * 4, 64);
      auto ssbAddr = builder.create<arith::AddIOp>(loc, ssbBaseAddr, ssbAddrOffset);
      Value ptr = builder.create<LLVM::IntToPtrOp>(loc, LLVM::LLVMPointerType::get(builder.getContext(), 11), ssbAddr);
      precomputedPtrs[groupIdx] = ptr;
    }
  }

  // 获取buffer指针的辅助函数
  // Cube: 直接从ssbufferPtrs按索引获取
  // Vector: 从预计算的指针表中获取
  auto getPtr = [&](OpBuilder &b, int groupIdx, int ptrSetIdx) -> Value {
    if (isAIC) {
      return ssbufferPtrs[ptrSetIdx][groupIdx];
    } else {
      return precomputedPtrs[groupIdx];
    }
  };

  // 条件组合辅助函数：将新条件与已有条件做AND运算
  auto combineCondition = [&](Value newCond) {
    if (conditions) {
      conditions = builder.create<arith::AndIOp>(loc, conditions, newCond);
    } else {
      conditions = newCond;
    }
  };

  // ========================================
  // 在ifOp前面插入输入/输出条件判断
  // ========================================
  builder.setInsertionPoint(ifOp);

  // 输入buffer条件: 控制变量 > 0 (表示buffer有数据可读)
  for (int inputGroupIdx : crossCoreInputValues) {
    Value cond = nullptr;
    if (isAIC) {
      // Cube模式: 同时检查vec0和vec1，都满足才继续
      Value vec0Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(), getPtr(builder, inputGroupIdx, 0));
      Value vec1Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(), getPtr(builder, inputGroupIdx, 1));
      Value vec0Cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, vec0Value, zeroConst);
      Value vec1Cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, vec1Value, zeroConst);
      cond = builder.create<arith::AndIOp>(loc, vec0Cond, vec1Cond);
    } else {
      // Vector模式: 只检查当前核对应的控制变量
      Value value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(), getPtr(builder, inputGroupIdx, 0));
      cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, value, zeroConst);
    }
    combineCondition(cond);
  }

  // 输出buffer条件: 控制变量 < 该组buffer的数量 (表示buffer有空位可写)
  for (int outputGroupIdx : crossCoreOutputValues) {
    // 统计该组所有output的数量
    int outputCount = 0;
    for (auto &entry : crossCoreBuffers[outputGroupIdx]) {
      outputCount += entry.second.size();
    }
    Value bufferNum = builder.create<arith::ConstantIntOp>(loc, outputCount, 32);
    Value cond = nullptr;
    if (isAIC) {
      Value vec0Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(), getPtr(builder, outputGroupIdx, 0));
      Value vec1Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(), getPtr(builder, outputGroupIdx, 1));
      Value vec0Cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, vec0Value, bufferNum);
      Value vec1Cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, vec1Value, bufferNum);
      cond = builder.create<arith::AndIOp>(loc, vec0Cond, vec1Cond);
    } else {
      Value value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(), getPtr(builder, outputGroupIdx, 0));
      cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, value, bufferNum);
    }
    combineCondition(cond);
  }

  // ========================================
  // 在ifOp内部(yield前)插入控制变量更新
  // ========================================
  Block* thenBlock = &ifOp.getThenRegion().front();
  auto yieldOp = cast<scf::YieldOp>(thenBlock->getTerminator());
  builder.setInsertionPoint(yieldOp);

  // 输入buffer: 消费后控制变量减1
  for (int inputGroupIdx : crossCoreInputValues) {
    if (isAIC) {
      Value vec0Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(), getPtr(builder, inputGroupIdx, 0));
      Value vec1Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(), getPtr(builder, inputGroupIdx, 1));
      Value vec0NewValue = builder.create<arith::SubIOp>(loc, vec0Value, oneConst);
      Value vec1NewValue = builder.create<arith::SubIOp>(loc, vec1Value, oneConst);
      builder.create<LLVM::StoreOp>(loc, vec0NewValue, getPtr(builder, inputGroupIdx, 0));
      builder.create<LLVM::StoreOp>(loc, vec1NewValue, getPtr(builder, inputGroupIdx, 1));
    } else {
      Value value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(), getPtr(builder, inputGroupIdx, 0));
      Value newValue = builder.create<arith::SubIOp>(loc, value, oneConst);
      builder.create<LLVM::StoreOp>(loc, newValue, getPtr(builder, inputGroupIdx, 0));
    }
  }

  // 输出buffer: 生产后控制变量加1
  for (int outputGroupIdx : crossCoreOutputValues) {
    if (isAIC) {
      Value vec0Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(), getPtr(builder, outputGroupIdx, 0));
      Value vec1Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(), getPtr(builder, outputGroupIdx, 1));
      Value vec0NewValue = builder.create<arith::AddIOp>(loc, vec0Value, oneConst);
      Value vec1NewValue = builder.create<arith::AddIOp>(loc, vec1Value, oneConst);
      builder.create<LLVM::StoreOp>(loc, vec0NewValue, getPtr(builder, outputGroupIdx, 0));
      builder.create<LLVM::StoreOp>(loc, vec1NewValue, getPtr(builder, outputGroupIdx, 1));
    } else {
      Value value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(), getPtr(builder, outputGroupIdx, 0));
      Value newValue = builder.create<arith::AddIOp>(loc, value, oneConst);
      builder.create<LLVM::StoreOp>(loc, newValue, getPtr(builder, outputGroupIdx, 0));
    }
  }

  return conditions;
}

Value AddIfControlsPass::getVarValue(scf::ForOp forOp, int varIndex) {
  if (!innerDepConds.count(forOp)) return Value();
  SmallVector<int> &innerDepIndices = innerDepConds[forOp];
  if (varIndex < (int) innerDepIndices.size()) {
    int argIdx = innerDepIndices[varIndex];
    return forOp.getRegionIterArgs()[argIdx];
  }
  return Value();
}

// 构建生产者组信息
SmallVector<OutputGroupInfo> AddIfControlsPass::buildOutputGroups(
  SmallVector<int> &intraCoreOutputValues,
  DenseMap<int, DenseMap<Value, SmallVector<Value> > > &intraCoreBuffers,
  DenseMap<int, Value> &idxToVar) {
  llvm::outs() << "[DEBUG] buildOutputGroups: 开始构建生产者组\n";
  llvm::outs() << "[DEBUG] buildOutputGroups: intraCoreOutputValues大小: " << intraCoreOutputValues.size() << "\n";
  llvm::outs() << "[DEBUG] buildOutputGroups: intraCoreOutputValues内容: ";
  for (int idx: intraCoreOutputValues) {
    llvm::outs() << idx << " ";
  }
  llvm::outs() << "\n";

  SmallVector<OutputGroupInfo> outputGroups;
  // 修改：使用更唯一的分组键，避免仅根据第一个生产者合并不同索引的组
  // 使用索引+消费者+生产者列表组合作为唯一标识

  for (int idx: intraCoreOutputValues) {
    llvm::outs() << "[DEBUG] buildOutputGroups: 处理索引 " << idx << "\n";

    auto bufferIt = intraCoreBuffers.find(idx);
    if (bufferIt == intraCoreBuffers.end()) {
      llvm::outs() << "[DEBUG] buildOutputGroups: 索引 " << idx << " 在intraCoreBuffers中未找到\n";
      continue;
    }

    auto varIt = idxToVar.find(idx);
    if (varIt == idxToVar.end()) {
      llvm::outs() << "[DEBUG] buildOutputGroups: 索引 " << idx << " 在idxToVar中未找到\n";
      continue;
    }
    Value var = varIt->second;
    llvm::outs() << "[DEBUG] buildOutputGroups: 索引 " << idx << " 对应的变量: " << var << "\n";

    llvm::outs() << "[DEBUG] buildOutputGroups: 索引 " << idx << " 在intraCoreBuffers中有 " << bufferIt->second.size() <<
        " 个条目\n";
    int entryCount = 0;
    for (auto &entry: bufferIt->second) {
      llvm::outs() << "[DEBUG] buildOutputGroups: 处理条目 " << entryCount++ << "\n";
      llvm::outs() << "[DEBUG] buildOutputGroups: 消费者: " << entry.first << "\n";

      SmallVector<Value> &outputs = entry.second;
      if (outputs.empty()) {
        llvm::outs() << "[DEBUG] buildOutputGroups: 生产者列表为空，跳过\n";
        continue;
      }

      llvm::outs() << "[DEBUG] buildOutputGroups: 生产者数量: " << outputs.size() << "\n";
      llvm::outs() << "[DEBUG] buildOutputGroups: 生产者列表: ";
      for (Value output: outputs) {
        llvm::outs() << output << " ";
      }
      llvm::outs() << "\n";

      // 修改：不再仅根据第一个生产者分组，而是为每个索引+消费者+生产者列表组合创建新组
      // 这样可以确保不同索引的生产者列表不会被错误合并
      bool flag = true;
      for (auto &outputGroup: outputGroups) {
        if (outputGroup.outputs == outputs) {
          outputGroup.inputVars.push_back(var);
          flag = false;
          llvm::outs() << "[DEBUG] buildOutputGroups: 加入消费者变量到已存在组 " << outputGroups.size() - 1 << "，索引: " << idx <<
              "，包含生产者: ";
          break;
        }
      }
      if (flag) {
        OutputGroupInfo info;
        info.outputs = outputs;
        info.inputVars.push_back(var);
        outputGroups.push_back(info);
        llvm::outs() << "[DEBUG] buildOutputGroups: 创建新组 " << outputGroups.size() - 1 << "，索引: " << idx <<
            "，包含生产者: ";
      }
      for (Value output: outputs) {
        llvm::outs() << output << " ";
      }
      llvm::outs() << "\n";
    }
  }

  llvm::outs() << "[DEBUG] buildOutputGroups: 完成构建，生产者组数量: " << outputGroups.size() << "\n";
  for (size_t i = 0; i < outputGroups.size(); ++i) {
    auto &group = outputGroups[i];
    llvm::outs() << "[DEBUG] buildOutputGroups: 组 " << i << " 信息:\n";
    llvm::outs() << "[DEBUG] buildOutputGroups:   生产者数量: " << group.outputs.size() << "\n";
    llvm::outs() << "[DEBUG] buildOutputGroups:   消费者变量数量: " << group.inputVars.size() << "\n";
    llvm::outs() << "[DEBUG] buildOutputGroups:   消费者变量: ";
    for (Value var: group.inputVars) {
      llvm::outs() << var << " ";
    }
    llvm::outs() << "\n";
    llvm::outs() << "[DEBUG] buildOutputGroups:   生产者列表: ";
    for (Value output: group.outputs) {
      llvm::outs() << output << " ";
    }
    llvm::outs() << "\n";
  }

  return outputGroups;
}



// 设置核内条件
// 变量更新操作将在 createNewIfOpWithBlocks 中正确创建
Value AddIfControlsPass::setIntraCoreCondition(
  ModuleOp module, scf::IfOp ifOp,
  DenseMap<int, DenseMap<Value, SmallVector<Value> > > &intraCoreBuffers,
  SmallVector<int> &intraCoreInputValues, SmallVector<int> &intraCoreOutputValues, DenseMap<int, Value> &idxToVar,
  DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  llvm::outs() << "[DEBUG] setIntraCoreCondition: 开始处理IfOp: " << ifOp << "\n";

  // 创建builder和位置信息
  OpBuilder builder(ifOp.getContext());
  builder.setInsertionPoint(ifOp);
  Location loc = ifOp.getLoc();

  // 存储所有生成的条件
  SmallVector<Value> conditions;
  DenseSet<Value> usedVarsSet;

  // 处理输入相关的条件
  if (!intraCoreInputValues.empty()) {
    llvm::outs() << "[DEBUG] setIntraCoreCondition: 开始收集输入变量并生成条件\n";
    Value zeroConst = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    for (int idx: intraCoreInputValues) {
      auto varIt = idxToVar.find(idx);
      if (varIt != idxToVar.end()) {
        Value var = varIt->second;
        // 检查是否有最新值，如果有，使用最新值
        Value varToUse = var;
        auto latestIt = controlVarToLatestValue.find(var);
        if (latestIt != controlVarToLatestValue.end()) {
          varToUse = latestIt->second;
          llvm::outs() << "[DEBUG] setIntraCoreCondition: 使用变量的最新值: " << var << " -> " << varToUse << "\n";
        }
        llvm::outs() << "[DEBUG] setIntraCoreCondition: 收集到输入变量: " << varToUse << " (原始变量: " << var << ", 索引: " << idx << ")\n";

        // 条件：varToUse > 0
        Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, varToUse, zeroConst);
        conditions.push_back(cond);
        llvm::outs() << "[DEBUG] setIntraCoreCondition: 生成输入条件: " << cond << " (变量: " << varToUse << ")\n";

        usedVarsSet.insert(var);  // 存储原始变量
        // 输入变量（消费者）：-1
        varUpdateTypes[var] = VarUpdateType::DEC;
        llvm::outs() << "[DEBUG] setIntraCoreCondition: 设置输入变量更新类型: " << var << " -> DEC (-1)\n";
      }
    }
    llvm::outs() << "[DEBUG] setIntraCoreCondition: 生成输入条件后 module：" << module << "\n";
    llvm::outs() << "[DEBUG] setIntraCoreCondition: 完成处理，共生成了 " << usedVarsSet.size() << " 个输入条件\n";
  }

  // 处理输出相关的条件
  if (!intraCoreOutputValues.empty()) {
    llvm::outs() << "[DEBUG] setIntraCoreCondition: 开始生成输出条件\n";
    SmallVector<OutputGroupInfo> outputGroups = buildOutputGroups(intraCoreOutputValues, intraCoreBuffers, idxToVar);
    for (auto &group: outputGroups) {
      int size = group.outputs.size();
      Value limitVal = builder.create<arith::ConstantIntOp>(loc, size, 32);
      for (Value var: group.inputVars) {
        // 检查是否有最新值，如果有，使用最新值
        Value varToUse = var;
        auto latestIt = controlVarToLatestValue.find(var);
        if (latestIt != controlVarToLatestValue.end()) {
          varToUse = latestIt->second;
          llvm::outs() << "[DEBUG] setIntraCoreCondition: 使用变量的最新值: " << var << " -> " << varToUse << "\n";
        }
        llvm::outs() << "[DEBUG] setIntraCoreCondition: 输出变量: " << varToUse << " (原始变量: " << var << ")\n";

        // 条件：varToUse < size
        Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, varToUse, limitVal);
        conditions.push_back(cond);
        llvm::outs() << "[DEBUG] setIntraCoreCondition: 生成输出条件: " << cond << " (变量: " << varToUse << ")\n";

        // 将变量添加到使用集合（存储原始变量）
        usedVarsSet.insert(var);
        // 输出变量（生产者）：+1
        varUpdateTypes[var] = VarUpdateType::INC;
        llvm::outs() << "[DEBUG] setIntraCoreCondition: 设置输出变量更新类型: " << var << " -> INC (+1)\n";
      }
    }
    llvm::outs() << "[DEBUG] setIntraCoreCondition: 生成输出条件后 module：" << module << "\n";
    llvm::outs() << "[DEBUG] setIntraCoreCondition: 完成生成，共生成了 " << outputGroups.size() << " 个输出条件\n";
  }

  // 合并所有核内条件
  llvm::outs() << "[DEBUG] setIntraCoreCondition: 开始合并条件\n";
  Value combinedCond;
  if (!conditions.empty()) {
    // 合并所有条件（使用AND操作）
    combinedCond = conditions[0];
    for (size_t i = 1; i < conditions.size(); ++i) {
      combinedCond = builder.create<arith::AndIOp>(loc, combinedCond, conditions[i]);
    }
  }
  llvm::outs() << "[DEBUG] setIntraCoreCondition: 合并所有条件后 module：" << module << "\n";
  llvm::outs() << "[DEBUG] setIntraCoreCondition: 完成合并，合并后的条件: " << combinedCond << "\n";

  // 将 usedVarsSet 存储到成员变量 currentUsedVars
  currentUsedVars.clear();
  for (Value var : usedVarsSet) {
    currentUsedVars.push_back(var);
  }
  llvm::outs() << "[DEBUG] setIntraCoreCondition: 存储了 " << currentUsedVars.size() << " 个使用的控制变量\n";
  for (size_t i = 0; i < currentUsedVars.size(); ++i) {
    llvm::outs() << "    控制变量[" << i << "]: " << currentUsedVars[i] << "\n";
  }

  llvm::outs() << "[DEBUG] setIntraCoreCondition: 完成处理，返回条件: " << combinedCond << "\n";

  return combinedCond;
}

// 更新 for 循环的 yield
void AddIfControlsPass::updateForOpYield(
  scf::ForOp forOp, scf::IfOp newIfOp, scf::IfOp oldIfOp,
  bool hasCounter, Value counter) {
  llvm::outs() << "[DEBUG] updateForOpYield: 开始更新 for 循环的 yield\n";
  llvm::outs() << "  控制变量数量: " << currentUsedVars.size() << "\n";
  llvm::outs() << "  是否有计数器: " << (hasCounter ? "是" : "否") << "\n";

  if (currentUsedVars.empty() && !hasCounter) {
    llvm::outs() << "  没有需要更新的控制变量和计数器，直接返回\n";
    return;
  }

  Location loc = newIfOp.getLoc();
  Block *forBody = forOp.getBody();
  auto yieldOp = cast<scf::YieldOp>(forBody->getTerminator());

  // 复制旧 yield 操作数
  SmallVector<Value> newYieldOperands(yieldOp.getOperands().begin(), yieldOp.getOperands().end());

  // 构建 for 循环迭代参数的索引映射（用于 O(1) 查找）
  DenseMap<Value, unsigned> iterArgToIndex;
  for (unsigned j = 0; j < forOp.getNumRegionIterArgs(); ++j) {
    iterArgToIndex[forOp.getRegionIterArgs()[j]] = j;
  }

  // 计算控制变量在 IfOp 结果中的起始位置
  size_t origResultCount = oldIfOp.getNumResults();
  size_t varStartIdx = origResultCount;

  // 用 IfOp 的结果替换控制变量
  for (size_t i = 0; i < currentUsedVars.size(); ++i) {
    Value var = currentUsedVars[i];
    size_t ifResultIdx = varStartIdx + i;
    auto it = iterArgToIndex.find(var);
    if (it != iterArgToIndex.end()) {
      newYieldOperands[it->second] = newIfOp.getResult(ifResultIdx);
    }
  }

  // 用 IfOp 的结果替换计数器
  if (hasCounter) {
    size_t counterResultIdx = varStartIdx + currentUsedVars.size();
    auto it = iterArgToIndex.find(counter);
    if (it != iterArgToIndex.end()) {
      newYieldOperands[it->second] = newIfOp.getResult(counterResultIdx);
    }
  }

  // 创建新的 yield 操作
  OpBuilder yieldBuilder(yieldOp);
  auto newYieldOp = yieldBuilder.create<scf::YieldOp>(loc, newYieldOperands);
  yieldOp.erase();
  llvm::outs() << "  新 yield 操作创建成功\n";

  // 更新全局映射 controlVarToLatestValue（后续的 replaceAllOldVarsWithLatest 会用到这个）
  for (size_t i = 0; i < currentUsedVars.size(); ++i) {
    Value var = currentUsedVars[i];
    Value newValue = newIfOp.getResult(origResultCount + i);
    controlVarToLatestValue[var] = newValue;
  }

  llvm::outs() << "[DEBUG] updateForOpYield: 完成更新 for 循环的 yield\n";
  llvm::outs() << "==================================================\n";
}

// 创建新 IfOp 并处理 then/else 块
// 核心逻辑：只要有控制变量或计数器，就必须有 yield 和 else 分支
// - 原来有结果的，在后面加上控制变量和计数器的值
// - 原来没有结果的，创建新的 yield 输出控制变量和计数器
scf::IfOp AddIfControlsPass::createNewIfOpWithBlocks(scf::IfOp oldIfOp, Value combinedCond,
                                                     DenseMap<Value, VarUpdateType> &varUpdateTypes,
                                                     bool hasCounter, Value counter,
                                                     Value step) {
  llvm::outs() << "[DEBUG] createNewIfOpWithBlocks: 开始创建新 IfOp\n";
  llvm::outs() << "  旧 IfOp: " << oldIfOp << "\n";
  llvm::outs() << "  合并后的条件: " << combinedCond << "\n";
  llvm::outs() << "  控制变量数量: " << currentUsedVars.size() << "\n";
  for (size_t i = 0; i < currentUsedVars.size(); ++i) {
    llvm::outs() << "    控制变量[" << i << "]: " << currentUsedVars[i] << "\n";
  }
  llvm::outs() << "  变量更新类型数量: " << varUpdateTypes.size() << "\n";
  for (auto &entry: varUpdateTypes) {
    const char *typeStr = "NONE";
    if (entry.second == VarUpdateType::INC) typeStr = "INC (+1)";
    else if (entry.second == VarUpdateType::DEC) typeStr = "DEC (-1)";
    llvm::outs() << "    " << entry.first << " → " << typeStr << "\n";
  }
  llvm::outs() << "  是否有计数器: " << (hasCounter ? "是" : "否") << "\n";
  if (hasCounter) {
    llvm::outs() << "  计数器: " << counter << "\n";
  }

  Location loc = oldIfOp.getLoc();
  OpBuilder builder(oldIfOp);

  // 判断是否需要 yield（有控制变量或计数器）
  bool needsYield = !currentUsedVars.empty() || hasCounter;

  // 检查旧 IfOp 是否有 else 块
  bool oldHasElse = oldIfOp.getElseRegion().hasOneBlock();

  // 检查旧 then 块的 yield
  Block &oldThenBlock = oldIfOp.getThenRegion().front();
  Operation *oldThenYieldOp = nullptr;
  SmallVector<Value> oldYieldOperands;
  if (!oldThenBlock.empty()) {
    Operation *lastOp = &oldThenBlock.back();
    if (isa<scf::YieldOp>(lastOp)) {
      oldThenYieldOp = lastOp;
      auto yieldOp = cast<scf::YieldOp>(lastOp);
      oldYieldOperands.assign(yieldOp.getOperands().begin(),
                              yieldOp.getOperands().end());
    }
  }
  llvm::outs() << "  旧 then 块有 yield: " << (oldThenYieldOp ? "是" : "否")
               << "，操作数数量: " << oldYieldOperands.size() << "\n";

  // 构建结果类型 = 原有结果 + 控制变量 + 计数器
  SmallVector<Type> resultTypes;
  for (Value result: oldIfOp.getResults()) {
    resultTypes.push_back(result.getType());
  }
  for (Value var: currentUsedVars) {
    resultTypes.push_back(var.getType());
  }
  if (hasCounter) {
    resultTypes.push_back(counter.getType());
  }
  llvm::outs() << "  结果类型数量: " << resultTypes.size() << "\n";

  // 创建新 IfOp
  scf::IfOp newIfOp;
  newIfOp = builder.create<scf::IfOp>(loc, resultTypes, combinedCond, true);
  llvm::outs() << "  创建带结果的 IfOp (hasElse=true): " << newIfOp << "\n";

  // 复制属性
  for (auto &attr: oldIfOp->getAttrs()) {
    newIfOp->setAttr(attr.getName(), attr.getValue());
  }

  // ========================================
  // 处理 then 块
  // ========================================
  Block &newThenBlock = newIfOp.getThenRegion().front();

  // 移动旧 then 块的操作（不包括 yield）
  for (Operation &op: llvm::make_early_inc_range(oldThenBlock)) {
    if (&op != oldThenYieldOp) {
      op.moveBefore(&newThenBlock, newThenBlock.end());
    }
  }

  // 创建新的 then yield
  OpBuilder thenBuilder(&newThenBlock, newThenBlock.end());
  SmallVector<Value> thenYieldOperands(oldYieldOperands.begin(), oldYieldOperands.end());

  // 添加控制变量（更新后的值）
  if (!currentUsedVars.empty()) {
    Value one = thenBuilder.create<arith::ConstantIntOp>(loc, 1, 32);
    for (Value var: currentUsedVars) {
      Value varToUse = var;
      auto latestIt = controlVarToLatestValue.find(var);
      if (latestIt != controlVarToLatestValue.end()) {
        varToUse = latestIt->second;
      }

      Value yieldVal = varToUse;
      auto it = varUpdateTypes.find(var);
      if (it != varUpdateTypes.end()) {
        if (it->second == VarUpdateType::DEC) {
          yieldVal = thenBuilder.create<arith::SubIOp>(loc, varToUse, one);
        } else if (it->second == VarUpdateType::INC) {
          yieldVal = thenBuilder.create<arith::AddIOp>(loc, varToUse, one);
        }
      }
      thenYieldOperands.push_back(yieldVal);
      llvm::outs() << "  then yield 控制变量: " << var << " → " << yieldVal << "\n";
    }
  }

  // 添加计数器（更新后的值）
  if (hasCounter) {
    Value newCounter = thenBuilder.create<arith::AddIOp>(loc, counter, step);
    thenYieldOperands.push_back(newCounter);
    llvm::outs() << "  then yield 计数器: " << counter << " → " << newCounter << "\n";
  }

  thenBuilder.create<scf::YieldOp>(loc, thenYieldOperands);
  llvm::outs() << "  创建 then yield，操作数数量: " << thenYieldOperands.size() << "\n";

  // ========================================
  // 处理 else 块（如果有控制变量或计数器，必须有 else 块）
  // ========================================
  if (needsYield) {
    Block &newElseBlock = newIfOp.getElseRegion().front();

    SmallVector<Value> oldElseYieldOperands;

    // 如果有旧 else 块，移动其内容
    if (oldHasElse) {
      Block &oldElseBlock = oldIfOp.getElseRegion().front();
      Operation *oldElseYieldOp = nullptr;

      if (!oldElseBlock.empty()) {
        Operation *lastOp = &oldElseBlock.back();
        if (isa<scf::YieldOp>(lastOp)) {
          oldElseYieldOp = lastOp;
          auto yieldOp = cast<scf::YieldOp>(lastOp);
          oldElseYieldOperands.assign(yieldOp.getOperands().begin(),
                                      yieldOp.getOperands().end());
        }
      }

      // 移动操作（不包括 yield）
      for (Operation &op: llvm::make_early_inc_range(oldElseBlock)) {
        if (&op != oldElseYieldOp) {
          op.moveBefore(&newElseBlock, newElseBlock.end());
        }
      }
    }

    // 创建新的 else yield
      OpBuilder elseBuilder(&newElseBlock, newElseBlock.end());
      SmallVector<Value> elseYieldOperands;

      // 替换旧 else yield 操作数中的旧变量
      for (Value operand : oldElseYieldOperands) {
        Value newOperand = operand;
        auto it = controlVarToLatestValue.find(operand);
        if (it != controlVarToLatestValue.end()) {
          newOperand = it->second;
        }
        elseYieldOperands.push_back(newOperand);
      }

      // 添加控制变量（不更新，使用最新值）
      for (Value var: currentUsedVars) {
        Value varToUse = var;
        auto it = controlVarToLatestValue.find(var);
        if (it != controlVarToLatestValue.end()) {
          varToUse = it->second;
        }
        elseYieldOperands.push_back(varToUse);
        llvm::outs() << "  else yield 控制变量（不更新）: " << var << " -> " << varToUse << "\n";
      }

      // 添加计数器（不更新，使用最新值）
      if (hasCounter) {
        Value counterToUse = counter;
        auto it = controlVarToLatestValue.find(counter);
        if (it != controlVarToLatestValue.end()) {
          counterToUse = it->second;
        }
        elseYieldOperands.push_back(counterToUse);
        llvm::outs() << "  else yield 计数器（不更新）: " << counter << " -> " << counterToUse << "\n";
      }

    elseBuilder.create<scf::YieldOp>(loc, elseYieldOperands);
    llvm::outs() << "  创建 else yield，操作数数量: " << elseYieldOperands.size() << "\n";
  } else if (oldHasElse) {
    // 没有控制变量和计数器，但有旧 else 块，只需要移动内容
    Block &oldElseBlock = oldIfOp.getElseRegion().front();
    Block &newElseBlock = newIfOp.getElseRegion().front();

    Operation *oldElseYieldOp = nullptr;
    if (!oldElseBlock.empty()) {
      Operation *lastOp = &oldElseBlock.back();
      if (isa<scf::YieldOp>(lastOp)) {
        oldElseYieldOp = lastOp;
      }
    }

    // 移动操作（不包括 yield）
    for (Operation &op: llvm::make_early_inc_range(oldElseBlock)) {
      if (&op != oldElseYieldOp) {
        op.moveBefore(&newElseBlock, newElseBlock.end());
      }
    }

    // 删除旧的 else yield（新 else 块已有自动创建的空 yield）
    if (oldElseYieldOp) {
      oldElseYieldOp->erase();
    }
  }

  // ========================================
  // 替换旧 IfOp 的使用（只替换原有结果）
  // ========================================
  for (size_t i = 0; i < oldIfOp.getNumResults(); ++i) {
    oldIfOp.getResult(i).replaceAllUsesWith(newIfOp.getResult(i));
    llvm::outs() << "  替换结果[" << i << "]: " << oldIfOp.getResult(i)
                 << " → " << newIfOp.getResult(i) << "\n";
  }

  llvm::outs() << "[DEBUG] createNewIfOpWithBlocks: 完成创建新 IfOp: " << newIfOp << "\n";
  return newIfOp;
}

// 合并三个条件：核间条件 + 核内条件 + 计数器条件
void AddIfControlsPass::combineConditions(ModuleOp module, Value crossCoreCond, Value intraCoreCond,
                                          scf::IfOp ifOp, scf::ForOp forOp,
                                          size_t &usedCounterNum,DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  llvm::outs() << "[DEBUG] combineConditions: 开始处理 IfOp: " << ifOp << "\n";

  Location loc = ifOp.getLoc();
  SmallVector<Value> validConditions;
  Value counter;
  bool hasCounter = false;

  // 1. 直接收集已创建的条件
  if (crossCoreCond) {
    validConditions.push_back(crossCoreCond);
    llvm::outs() << "  收集核间条件: " << crossCoreCond << "\n";
  }
  if (intraCoreCond) {
    validConditions.push_back(intraCoreCond);
    llvm::outs() << "  收集核内条件: " << intraCoreCond << "\n";
  }

  // 2. 为 IfOp 分配计数器并创建计数器条件
  if (blockCounters.count(forOp)) {
    SmallVector<int> &counterIndices = blockCounters[forOp];
    llvm::outs() << "  可用计数器索引数量: " << counterIndices.size() << "\n";

    if (cntArgs.count(ifOp)) {
      counter = cntArgs[ifOp];
      hasCounter = true;
      llvm::outs() << "  已存在计数器: " << counter << "\n";
    } else if (usedCounterNum < counterIndices.size()) {
      int argIdx = counterIndices[usedCounterNum];
      counter = forOp.getRegionIterArgs()[argIdx];
      hasCounter = true;
      cntArgs[ifOp] = counter;
      usedCounterNum++;
      llvm::outs() << "  分配计数器: " << counter << ", index=" << usedCounterNum - 1 << "\n";
    }

    // 创建计数器条件
    if (hasCounter) {
      OpBuilder builder(ifOp);
      Value upperBound = forOp.getUpperBound();
      // 检查计数器是否有最新值，如果有，使用最新值
      Value counterToUse = counter;
      auto latestIt = controlVarToLatestValue.find(counter);
      if (latestIt != controlVarToLatestValue.end()) {
        counterToUse = latestIt->second;
        llvm::outs() << "  使用计数器的最新值: " << counter << " -> " << counterToUse << "\n";
      }
      // 比较 counterToUse 和 upperBound
      Value counterCond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, counterToUse, upperBound);
      validConditions.push_back(counterCond);
      llvm::outs() << "  创建计数器条件: " << counterCond << " (计数器: " << counterToUse << ")\n";
      llvm::outs() << "[DEBUG] combineConditions: 创建计数器条件后 module：" << module << "\n";
    }
  }
  //核间+核内+计数器条件都为空时，不创建新的IfOp
  if (validConditions.empty()) {
    return;
  }
  // 3. 合并所有条件
  Value combinedCond;
  if (validConditions.size() != 0) {
    OpBuilder builder(ifOp);
    combinedCond = validConditions[0];
    for (size_t i = 1; i < validConditions.size(); ++i) {
      combinedCond = builder.create<arith::AndIOp>(loc, combinedCond, validConditions[i]);
    }
    llvm::outs() << "  合并后条件: " << combinedCond << "\n";
  }
  llvm::outs() << "[DEBUG] combineConditions: 组合所有条件后 module：" << module << "\n";

  // 4. 创建新 IfOp
  llvm::outs() << "[DEBUG] combineConditions: 创建新 IfOp\n";
  scf::IfOp newIfOp = createNewIfOpWithBlocks(ifOp, combinedCond, varUpdateTypes, hasCounter, counter, forOp.getStep());
  llvm::outs() << "[DEBUG] combineConditions: 创建新 IfOp 后 module：" << module << "\n";

  // 6. 更新计数器映射
  if (hasCounter) {
    cntArgs.erase(ifOp);
    cntArgs[newIfOp] = counter;
  }

  // 7. 更新 for 循环的 yield
  updateForOpYield(forOp, newIfOp, ifOp, hasCounter, counter);

  // 8. 删除旧 IfOp
  ifOp.erase();

  llvm::outs() << "[DEBUG] combineConditions: 完成\n";
  llvm::outs() << "==================================================\n";
  llvm::outs() << "[DEBUG] combineConditions: 最终 module：" << module << "\n";
}

void AddIfControlsPass::collectDependencyBuffers(
  scf::ForOp forOp,
  DenseMap<int, DenseMap<Value, SmallVector<Value> > > &crossCoreBuffers,
  DenseMap<int, DenseMap<Value, SmallVector<Value> > > &intraCoreBuffers) {
  crossCoreBuffers.clear();
  intraCoreBuffers.clear();

  int crossCoreIdx = 0;
  for (auto &entry: crossCoreDependentMap) {
    crossCoreBuffers[crossCoreIdx][entry.first] = entry.second;
    crossCoreIdx++;
  }

  if (intraCoreDependentMap.count(forOp)) {
    auto &forOpDeps = intraCoreDependentMap[forOp];
    int intraCoreIdx = 0;
    for (auto &entry: forOpDeps) {
      intraCoreBuffers[intraCoreIdx][entry.first] = entry.second;
      intraCoreIdx++;
    }
  }
}
// 扩展crossCoreBuffers，添加通过annotation.mark标记的等价value到SmallVector中
// 用于处理新MLIR中fixpipe/copy操作的buffer通过annotation.mark标记为等价的情况
// 参数: module - 用于遍历所有annotation.mark操作
//       crossCoreBuffers - 原始的核间buffer依赖映射
// 返回: 新的DenseMap<int, DenseMap<Value, SmallVector<Value>>>，包含等价value
DenseMap<int, DenseMap<Value, SmallVector<Value>>> AddIfControlsPass::extendCrossCoreBuffersWithEquivalentValues(
    ModuleOp module,
    DenseMap<int, DenseMap<Value, SmallVector<Value>>> crossCoreBuffers)
{

  // 复制原始crossCoreBuffers
  DenseMap<int, DenseMap<Value, SmallVector<Value>>> extendedCrossCoreBuffers;
  for (auto &entry : crossCoreBuffers) {
    int groupIdx = entry.first;
    for (auto &entry2 : entry.second) {
      extendedCrossCoreBuffers[groupIdx][entry2.first] = entry2.second;
    }
  }

  // 构建annotation.mark等价映射: tightly_coupled_buffer值 -> 所有被标记为该值的value列表
  DenseMap<int, SmallVector<Value>> tightlyCoupledBufferGroups;
  if (module) {
    module.walk([&](Operation *op) {
      // 检查是否是annotation.mark操作
      // if (op->getName().getStringRef() == "annotation.mark") {
      if (isa<annotation::MarkOp>(op)) {
        llvm::outs() << "find markop: " << *op << "\n";
        if (op->getNumOperands() >= 1) {
          llvm::outs() << "1111\n";
          Value markedValue = op->getOperand(0);
          // 获取tightly_coupled_buffer属性
          if (auto tcbAttr = op->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>("hivm.tightly_coupled_buffer")) {
            llvm::outs() << "find mark value: " << markedValue << "\n";
            auto id = tcbAttr.getId();
            if (id.has_value()) {
              int tcb = id.value();
              tightlyCoupledBufferGroups[tcb].push_back(markedValue);
            }
          }
        }
      }
    });
  }

  // 打印 tightlyCoupledBufferGroups
  llvm::outs() << "  tightlyCoupledBufferGroups size: " << tightlyCoupledBufferGroups.size() << "\n";
  for (auto &entry : tightlyCoupledBufferGroups) {
    int tcb = entry.first;
    llvm::outs() << "    tcb=" << tcb << " values: ";
    for (Value v : entry.second) {
      llvm::outs() << v << " ";
    }
    llvm::outs() << "\n";
  }

  // 扩展：对于已在extendedCrossCoreBuffers中的value，找到其等价value（相同tightly_coupled_buffer）
  // 也push到对应group的SmallVector中
  llvm::outs() << "=== extendCrossCoreBuffersWithEquivalentValues ===\n";
  for (auto &entry : extendedCrossCoreBuffers) {
    int groupIdx = entry.first;
    for (auto &entry2 : entry.second) {
      SmallVector<Value> &values = entry2.second;
      // 对于SmallVector中的每个value，查找其等价value
      for (Value v : values) {
        // 查找该value是否被annotation.mark标记
        for (auto &tcbEntry : tightlyCoupledBufferGroups) {
          SmallVector<Value> &tcbValues = tcbEntry.second;
          // 检查v是否在这个tcb组中
          for (Value tcbV : tcbValues) {
            if (tcbV == v) {
              // 找到了，将同组其他value也push到SmallVector中
              for (Value equivValue : tcbValues) {
                if (equivValue != v && !llvm::is_contained(values, equivValue)) {
                  values.push_back(equivValue);
                  llvm::outs() << "  group " << groupIdx << ": add equivalent " << equivValue << "\n";
                }
              }
              break;
            }
          }
        }
      }
    }
  }

  llvm::outs() << "  extendedCrossCoreBuffers groups: " << extendedCrossCoreBuffers.size() << "\n";
  return extendedCrossCoreBuffers;
}

// 获取ifOp的输入输出依赖组别索引
// crossCoreBuffers/intraCoreBuffers: 每组依赖关系 {groupIdx -> {inputValue: [outputValues...]}}
// crossCoreBufferToGroupExtended: 扩展后的buffer value -> group index映射（包含等价value）
// 返回值: 当前ifOp使用到的依赖组别索引，分为输入(消费者)和输出(生产者)
// 特殊情况处理: 一个output被多个input使用时，output对应多个group索引
void AddIfControlsPass::getInputOutputValues(scf::IfOp ifOp,
                                             DenseMap<int, DenseMap<Value, SmallVector<Value>>> crossCoreBuffers,
                                             DenseMap<int, DenseMap<Value, SmallVector<Value>>> intraCoreBuffers,
                                             SmallVector<int> &crossCoreInputValues,
                                             SmallVector<int> &crossCoreOutputValues,
                                             SmallVector<int> &intraCoreInputValues,
                                             SmallVector<int> &intraCoreOutputValues)
{
  DenseSet<int> crossCoreInputSet;
  DenseSet<int> crossCoreOutputSet;
  DenseSet<int> intraCoreInputSet;
  DenseSet<int> intraCoreOutputSet;

  // crossCoreBufferToGroup: buffer value -> group index (一对一映射)
  DenseMap<Value, int> crossCoreBufferToGroup;
  // intraCoreInputToGroup: input buffer value -> group index (一对一映射)
  DenseMap<Value, int> intraCoreInputToGroup;
  // intraCoreOutputToGroups: output buffer value -> 所有使用该output的group indices (一对多映射)
  // 场景: 同一个output被多个input使用时，需要返回所有相关的group索引
  DenseMap<Value, SmallVector<int>> intraCoreOutputToGroups;

  // 构建crossCore buffer到group索引的映射
  for (auto &entry : crossCoreBuffers) {
    int groupIdx = entry.first;
    for (auto &entry2 : entry.second) {
      for (Value v : entry2.second) {
        crossCoreBufferToGroup[v] = groupIdx;
      }
    }
  }

  // 构建intraCore buffer到group索引的映射
  // 注意: 同一个output可能被多个input使用，因此在intraCoreOutputToGroups中需要存储多个group索引
  for (auto &entry : intraCoreBuffers) {
    int groupIdx = entry.first;
    for (auto &entry2 : entry.second) {
      Value input = entry2.first;
      SmallVector<Value> outputs = entry2.second;
      // input buffer直接映射到对应的group索引
      intraCoreInputToGroup[input] = groupIdx;
      // output buffer需要映射到所有使用该output的group索引
      for (Value output : outputs) {
        intraCoreOutputToGroups[output].push_back(groupIdx);
      }
    }
  }

  // 遍历ifOp内部所有操作，确定输入输出的group索引
  ifOp.walk([&](Operation *op) {
    if (op == ifOp)
      return WalkResult::advance();

    // 处理FixpipeOp、CopyOp或bufferization写入op：第一个操作数作为输入，第二个操作数作为输出
    bool isFixpipeOrCopy = dyn_cast<hivm::FixpipeOp>(op) || dyn_cast<hivm::CopyOp>(op);
    bool isBufferizationWrite = dyn_cast<bufferization::MaterializeInDestinationOp>(op);

    if (isFixpipeOrCopy || isBufferizationWrite) {
      // 第一个操作数作为输入
      llvm::outs() << "find fixpipe or copy\n";
      Value insVal = op->getOperands()[0];
        llvm::outs() << "insVal: " << insVal << "\n";
      if (crossCoreBufferToGroup.count(insVal)) {
        crossCoreInputSet.insert(crossCoreBufferToGroup[insVal]);
        llvm::outs() << "is cross input\n";
      } else if (intraCoreInputToGroup.count(insVal)) {
        intraCoreInputSet.insert(intraCoreInputToGroup[insVal]);
        llvm::outs() << "is intra input\n";
      }

      // 第二个操作数作为输出
      Value outsVal = op->getOperands()[1];
        llvm::outs() << "outsVal: " << outsVal << "\n";
      if (crossCoreBufferToGroup.count(outsVal)) {
        llvm::outs() << "is cross output\n";
        crossCoreOutputSet.insert(crossCoreBufferToGroup[outsVal]);
      } else if (intraCoreOutputToGroups.count(outsVal)) {
      // output可能属于多个依赖组，需要返回所有相关的group索引
        llvm::outs() << "is intra output\n";
        for (int idx : intraCoreOutputToGroups[outsVal]) {
          intraCoreOutputSet.insert(idx);
        }
      }
      return WalkResult::advance();
    } else {
      // 普通操作：操作数作为输入
      for (Value operand : op->getOperands()) {
        if (crossCoreBufferToGroup.count(operand))
          crossCoreInputSet.insert(crossCoreBufferToGroup[operand]);
        if (intraCoreInputToGroup.count(operand))
          intraCoreInputSet.insert(intraCoreInputToGroup[operand]);
      }
    }
    // // 为了适配上游不完整的mlir的临时修改
    // for (auto opResult : op->getResults()) {
    //   if (crossCoreBufferToGroup.count(opResult))
    //     crossCoreOutputSet.insert(crossCoreBufferToGroup[opResult]);
    //   // producer可能属于多个依赖组，需要返回所有相关的group索引
    //   if (intraCoreProducerToGroups.count(opResult)) {
    //     for (int idx : intraCoreProducerToGroups[opResult]) {
    //       intraCoreOutputSet.insert(idx);
    //     }
    //   }
    // }
    return WalkResult::advance();
  });

  // ifOp 的 thenYield op 所 yield 的值作为输出
  scf::YieldOp thenYield = ifOp.thenYield();
  for (Value yieldVal : thenYield.getOperands()) {
    if (crossCoreBufferToGroup.count(yieldVal))
      crossCoreOutputSet.insert(crossCoreBufferToGroup[yieldVal]);
    // output可能属于多个依赖组，需要返回所有相关的group索引
    if (intraCoreOutputToGroups.count(yieldVal)) {
      for (int idx : intraCoreOutputToGroups[yieldVal]) {
        intraCoreOutputSet.insert(idx);
      }
    }
  }

  // 转换结果为SmallVector
  crossCoreInputValues.assign(crossCoreInputSet.begin(), crossCoreInputSet.end());
  crossCoreOutputValues.assign(crossCoreOutputSet.begin(), crossCoreOutputSet.end());
  intraCoreInputValues.assign(intraCoreInputSet.begin(), intraCoreInputSet.end());
  intraCoreOutputValues.assign(intraCoreOutputSet.begin(), intraCoreOutputSet.end());
}

void AddIfControlsPass::initTestData(ModuleOp module) {
  module.walk([&](memref::AllocOp allocOp) {
    Value result = allocOp.getResult();
    if (isa<MemRefType>(result.getType())) {
      auto memrefType = dyn_cast<MemRefType>(result.getType());
      if (isa<hivm::AddressSpaceAttr>(memrefType.getMemorySpace())) {
        auto space = dyn_cast<hivm::AddressSpaceAttr>(memrefType.getMemorySpace()).getAddressSpace();
        if (space == hivm::AddressSpace::UB || space == hivm::AddressSpace::L1) {
          crossCoreDependentMap[result] = {result};
        }
      }
    }
  });

  scf::ForOp parentFor;
  DenseMap<Value, SmallVector<Value>> tempMap;
  module.walk([&](arith::SelectOp selectOp) {
    auto result = selectOp.getResult();
    auto operands = selectOp.getOperands();
    if (operands.size() >= 3) {
      Value operand1 = operands[1];
      Value operand2 = operands[2];
      if (auto arg1 = dyn_cast<OpResult>(operand1)) {
        if (arg1.getResultNumber() == 0) {
          if (auto arg2 = dyn_cast<OpResult>(operand2)) {
            if (arg2.getResultNumber() == 4 || arg2.getResultNumber() == 6) {
              parentFor = dyn_cast<scf::ForOp>(selectOp->getParentOp()->getParentOp());
              if (parentFor) {
                tempMap[result] = {operand1, operand2};
                llvm::outs() << "find forop\n";
                llvm::outs() << parentFor << "\n";
              } else {
                llvm::outs() << "can not find forop\n";
              }
            }
          }
        }
      }
    }
  });
  intraCoreDependentMap[parentFor] = tempMap;
  // 为带有 {ssbuffer.main_loop} 属性的 scf::ForOp 初始化 blockCounters 和 innerDepConds
  int forOpCounter = 0;
  module.walk([&](scf::ForOp forOp) {
    // 检查 ForOp 是否带有 ssbuffer.main_loop 属性
    if (forOp->hasAttr("ssbuffer.main_loop")) {
      unsigned numIterArgs = forOp.getNumRegionIterArgs();
      llvm::outs() << "[DEBUG] 处理带有 ssbuffer.main_loop 属性的 ForOp: " << forOp << "\n";
      llvm::outs() << "  迭代参数数量: " << numIterArgs << "\n";

      // 为当前 ForOp 分配 blockCounters 和 innerDepConds
      SmallVector<int> counters;
      SmallVector<int> depConds;

      if (forOpCounter == 0) {
        // 第一个 forop - 基于 test.mlir 实际结构
        // iter_args: 13个参数
        // 标量索引: 1, 2, 3, 4, 5, 6, 8, 9, 11, 12
        llvm::outs() << "[DEBUG] 第一个 forop 迭代参数数量: " << numIterArgs << "\n";

        // 根据实际可用的标量索引分配
        // 依赖条件变量innerDepConds: 选择标量索引1, 2, 3, 4
         depConds = {1, 2, 3, 4};
        // 计数器变量blockCounters: 选择标量索引5, 6, 8
         counters = {5, 6, 8};




      } else if (forOpCounter == 1) {
        // 第二个 forop - 基于 test.mlir 实际结构
        // iter_args: 8个参数
        // 所有都是i64标量: 0, 1, 2, 3, 4, 5, 6, 7
        llvm::outs() << "[DEBUG] 第二个 forop 迭代参数数量: " << numIterArgs << "\n";

        // 根据实际可用的标量索引分配
        // 依赖条件变量innerDepConds: 前5个标量索引0, 1, 2, 3, 4
        depConds = {0, 1};
        // 计数器变量blockCounters: 后3个标量索引5, 6, 7
        counters = {2, 3};

      }

      // 分配索引
      blockCounters[forOp] = counters;
      innerDepConds[forOp] = depConds;
      forOpCounter++;
    }
  });
}

void AddIfControlsPass::updateIfConds(ModuleOp module) {

    SmallVector<SmallVector<Value>> ssbufferPtrs = initSSBuffer(module);
    // 输出 blockCounters 和 innerDepConds 完整内容
    llvm::outs() << "\n=== blockCounters 完整内容 ===\n";
    for (auto &entry: blockCounters) {
      scf::ForOp forOp = entry.first;
      SmallVector<int> &counters = entry.second;
      llvm::outs() << "ForOp @ " << forOp << "\n";
      llvm::outs() << "  Counters: [";
      for (size_t i = 0; i < counters.size(); i++) {
        llvm::outs() << counters[i];
        if (i < counters.size() - 1) llvm::outs() << ", ";
      }
      llvm::outs() << "]\n";
    }

    llvm::outs() << "\n=== innerDepConds 完整内容 ===\n";
    for (auto &entry: innerDepConds) {
      scf::ForOp forOp = entry.first;
      SmallVector<int> &deps = entry.second;
      llvm::outs() << "ForOp @ " << forOp << "\n";
      llvm::outs() << "  Dependencies: [";
      for (size_t i = 0; i < deps.size(); i++) {
        llvm::outs() << deps[i];
        if (i < deps.size() - 1) llvm::outs() << ", ";
      }
      llvm::outs() << "]\n";
    }
    // 遍历模块中所有的 scf::ForOp
    module.walk([&](scf::ForOp forOp) {
        // 检查 forOp 是否带有 ssbuffer.main_loop 属性
        if (forOp->hasAttr("ssbuffer.main_loop")) {
            // 清空全局映射，确保不同 for 循环之间的控制变量不会混淆
            controlVarToLatestValue.clear();

            // 当前for循环内的所有核内依赖，map{idx: {input_value: [output1, output2, ...]}, ...}
            DenseMap<int, DenseMap<Value, SmallVector<Value>>> crossCoreBuffers;
            DenseMap<int, DenseMap<Value, SmallVector<Value>>> intraCoreBuffers;
            collectDependencyBuffers(forOp, crossCoreBuffers, intraCoreBuffers);

            // 扩展crossCoreBuffers，添加通过annotation.mark标记的等价value
            DenseMap<int, DenseMap<Value, SmallVector<Value>>> extendedCrossCoreBuffers =
                extendCrossCoreBuffersWithEquivalentValues(module, crossCoreBuffers);
            llvm::outs() << "extendedCrossCoreBuffers.size(): " << extendedCrossCoreBuffers.size() << "\n";
            for (auto &entry : extendedCrossCoreBuffers) {
                llvm::outs() << "  extendedCrossCoreBuffers group " << entry.first << ": ";
                for (auto &entry2 : entry.second) {
                    llvm::outs() << "input=" << entry2.first << ", outputs=[";
                    for (auto v : entry2.second) {
                        llvm::outs() << v << ", ";
                    }
                    llvm::outs() << "]";
                }
                llvm::outs() << "\n";
            }

            llvm::outs() << "=== getInputOutputValues Input ===\n";
            llvm::outs() << "crossCoreBuffers.size(): " << crossCoreBuffers.size() << "\n";
            llvm::outs() << "intraCoreBuffers.size(): " << intraCoreBuffers.size() << "\n";
            for (auto &entry : crossCoreBuffers) {
                llvm::outs() << "  crossCore group " << entry.first << ": ";
                for (auto &entry2 : entry.second) {
                    llvm::outs() << "input=" << entry2.first << ", outputs=[";
                    for (auto v : entry2.second) {
                        llvm::outs() << v << ", ";
                    }
                    llvm::outs() << "]";
                }
                llvm::outs() << "\n";
            }
            for (auto &entry : intraCoreBuffers) {
                llvm::outs() << "  intraCore group " << entry.first << ": ";
                for (auto &entry2 : entry.second) {
                    llvm::outs() << "input=" << entry2.first << ", outputs=[";
                    for (auto v : entry2.second) {
                        llvm::outs() << v << ", ";
                    }
                    llvm::outs() << "]";
                }
                llvm::outs() << "\n";
            }

            // 为所有消费者 idx 分配变量
            DenseMap<int, Value> idxToVar;
            int varIdx = 0;
            for (auto &entry : intraCoreBuffers) {
                int idx = entry.first;
                Value var = getVarValue(forOp, varIdx);
                if (var) {
                    idxToVar[idx] = var;
                    varIdx++;
                }
            }

            // 遍历 forOp 内部区域中的所有 scf::IfOp
            size_t usedCounterNum = 0;  // 移动到这里，在同一 forOp 下累积计数
            forOp.walk([&](scf::IfOp ifOp) {
                // 检查 ifOp 是否带有 ssbuffer.if 属性
                if (ifOp->hasAttr("ssbuffer.if")) {
                    llvm::outs() << "ifOp: " << ifOp << "\n";
                    SmallVector<int> crossCoreInputValues;
                    SmallVector<int> crossCoreOutputValues;
                    // 当前ifop内涉及到的核内消费者依赖的idx列表
                    SmallVector<int> intraCoreInputValues;
                    // 当前ifop内涉及到的核内生产者依赖的idx列表
                    SmallVector<int> intraCoreOutputValues;

                    getInputOutputValues(ifOp, extendedCrossCoreBuffers, intraCoreBuffers,
                                        crossCoreInputValues, crossCoreOutputValues,
                                        intraCoreInputValues, intraCoreOutputValues);

                    llvm::outs() << "=== getInputOutputValues Output ===\n";
                    llvm::outs() << "ifOp: " << ifOp << "\n";
                    llvm::outs() << "crossCoreInputValues: [";
                    for (auto v : crossCoreInputValues) llvm::outs() << v << ", ";
                    llvm::outs() << "]\n";
                    llvm::outs() << "crossCoreOutputValues: [";
                    for (auto v : crossCoreOutputValues) llvm::outs() << v << ", ";
                    llvm::outs() << "]\n";
                    llvm::outs() << "intraCoreInputValues: [";
                    for (auto v : intraCoreInputValues) llvm::outs() << v << ", ";
                    llvm::outs() << "]\n";
                    llvm::outs() << "intraCoreOutputValues: [";
                    for (auto v : intraCoreOutputValues) llvm::outs() << v << ", ";
                    llvm::outs() << "]\n";

                    // 设置核间条件
                    Value crossCoreCond = setCrossCoreCondition(crossCoreInputValues,
                                                                crossCoreOutputValues,
                                                                crossCoreBuffers, ifOp, ssbufferPtrs);

                    // // 设置核内条件
                    // 用于记录变量更新类型（var -> INC/DEC）
                    DenseMap<Value, VarUpdateType> varUpdateTypes;
                    Value intraCoreCond = setIntraCoreCondition(module, ifOp, intraCoreBuffers, intraCoreInputValues,
                                                                intraCoreOutputValues, idxToVar, varUpdateTypes);

                    // // 合并条件并更新 IfOp
                    combineConditions(module, crossCoreCond, intraCoreCond, ifOp, forOp, usedCounterNum, varUpdateTypes);
                }
            });
        }
    });
}

// 构造 cntArgs: 找到所有带有 ssbuffer.if 的 scf::IfOp，根据条件找到对应的计数器值
// 逻辑: ifOp.condition的definingOp的第二个操作数, 这个操作数的definingOp是cmpi, cmpi的第一个操作数就是计数器
void constructCntArgs(ModuleOp module, DenseMap<scf::IfOp, Value> &cntArgs) {
  cntArgs.clear();

  module.walk([&](scf::IfOp ifOp) {
    if (!ifOp->hasAttr("ssbuffer.if")) return;

    llvm::outs() << "ifOp: " << ifOp << "\n";
    Value cond = ifOp.getCondition();
    llvm::outs() << "  cond: " << cond << "\n";

    Operation* condOp = cond.getDefiningOp();
    if (!condOp) return;

    Value cntVal = nullptr;
    // andi/or的第二个操作数
    if (mlir::isa<arith::AndIOp>(condOp) || mlir::isa<arith::OrIOp>(condOp)) {
      Value secondOperand = condOp->getOperand(1);
      Operation* secondOp = secondOperand.getDefiningOp();
      if (secondOp && mlir::isa<arith::CmpIOp>(secondOp)) {
        // cmpi的第一个操作数就是计数器
        cntVal = secondOp->getOperand(0);
      }
    }

    llvm::outs() << "  cntVal: " << cntVal << "\n";

    if (cntVal) {
      cntArgs[ifOp] = cntVal;
    }
  });
}

// 计算乘积因子 factor = requiredBuffers / x
// 对于 for 循环内的 if 计算块编号 1, 2, 3, ..., n
// 如果计算块 m 依赖计算块 n 的结果（m > n），需要 (m-n+1) 个核内 buffer
// 如果核内 buffer 仅提供 x 个，则 factor = (m-n+1) / x
// 返回具有最大 factor 的依赖的 (requiredBuffers, x)
// 使用整数比较：比较 a/b 和 c/d 等价于比较 a*d 和 c*b
std::pair<int, int> calculateFactor(scf::ForOp forOp,
                                     DenseMap<scf::ForOp, DenseMap<Value, SmallVector<Value>>> &intraCoreDependentMap)
{
  // 默认返回 (1, 1)，即 factor = 1
  int maxRequiredBuffers = 1;
  int maxX = 1;

  // 1. 找到 for 循环内的所有 if 计算块，并按顺序编号
  SmallVector<scf::IfOp> ifOps;
  DenseMap<Operation*, int> ifOpIndex;  // if op 到编号的映射（编号从 1 开始）
  int index = 1;

  forOp.walk([&](scf::IfOp ifOp) {
    if (ifOp->hasAttr("ssbuffer.if")) {
      ifOps.push_back(ifOp);
      ifOpIndex[ifOp.getOperation()] = index++;
    }
  });

  llvm::outs() << "[calculateFactor] forOp 内的 if 计算块数量: " << ifOps.size() << "\n";

  // 如果没有 if 计算块，返回 (1, 1)
  if (ifOps.empty()) {
    llvm::outs() << "[calculateFactor] 没有 if 计算块，返回 (1, 1)\n";
    return {1, 1};
  }

  // 2. 从 intraCoreDependentMap 获取核内依赖
  if (!intraCoreDependentMap.count(forOp)) {
    llvm::outs() << "[calculateFactor] intraCoreDependentMap 中没有该 forOp，返回 (1, 1)\n";
    return {1, 1};
  }

  auto &deps = intraCoreDependentMap[forOp];
  llvm::outs() << "[calculateFactor] 核内依赖数量: " << deps.size() << "\n";

  // 如果没有核内依赖，返回 (1, 1)
  if (deps.empty()) {
    llvm::outs() << "[calculateFactor] 没有核内依赖，返回 (1, 1)\n";
    return {1, 1};
  }

  // 3. 对于每个核内依赖，计算 requiredBuffers 和 x，找到最大 factor
  // 使用整数比较：比较 requiredBuffers1/x1 和 requiredBuffers2/x2
  // 等价于比较 requiredBuffers1 * x2 和 requiredBuffers2 * x1

  for (auto &entry : deps) {
    Value consumerResult = entry.first;  // 消费者 if 计算块 m 的结果
    SmallVector<Value> producerBuffers = entry.second;  // 生产者 if 计算块 n 产生的 buffer 列表
    int x = producerBuffers.size();  // buffer 数量

    llvm::outs() << "[calculateFactor] 处理依赖: consumerResult = " << consumerResult
                 << ", buffer数量 x = " << x << "\n";

    // 找到 consumerResult 是哪个 if 计算块产生的（编号 m）- 消费者
    Operation *consumerDefOp = consumerResult.getDefiningOp();
    if (!consumerDefOp) {
      llvm::outs() << "[calculateFactor] consumerResult 不是由 op 定义，跳过\n";
      continue;
    }

    // 找到包含 consumerDefOp 的 if 计算块
    scf::IfOp consumerIfOp = consumerDefOp->getParentOfType<scf::IfOp>();
    if (!consumerIfOp || !consumerIfOp->hasAttr("ssbuffer.if")) {
      llvm::outs() << "[calculateFactor] consumerResult 的定义 op 不在 if 计算块内，跳过\n";
      continue;
    }

    int m = ifOpIndex[consumerIfOp.getOperation()];
    llvm::outs() << "[calculateFactor] consumerResult 由 if 计算块 #" << m << " 产生（消费者）\n";

    // 找到 producerBuffers 是哪个 if 计算块产生的（编号 n）- 生产者
    // 取第一个 buffer 来确定生产者 if 计算块
    if (producerBuffers.empty()) {
      llvm::outs() << "[calculateFactor] producerBuffers 为空，跳过\n";
      continue;
    }

    Value firstProducerBuffer = producerBuffers[0];
    llvm::outs() << "firstProducerBuffer : " << firstProducerBuffer << "\n";
    Operation *producerDefOp = firstProducerBuffer.getDefiningOp();
    if (!producerDefOp) {
      llvm::outs() << "[calculateFactor] producerBuffer 不是由 op 定义，跳过\n";
      continue;
    }

    // 找到包含 producerDefOp 的 if 计算块
    scf::IfOp producerIfOp = producerDefOp->getParentOfType<scf::IfOp>();

    // // 临时逻辑
    // if (isa<scf::IfOp>(producerDefOp)) {
    //   producerIfOp = dyn_cast<scf::IfOp>(producerDefOp);
    // }
    // // === 临时逻辑结束 ===

    if (!producerIfOp || !(producerIfOp->hasAttr("ssbuffer.if"))) {
      llvm::outs() << "[calculateFactor] producerBuffer 的定义 op 不在 if 计算块内，跳过\n";
      continue;
    }

    int n = ifOpIndex[producerIfOp.getOperation()];
    llvm::outs() << "[calculateFactor] producerBuffer 由 if 计算块 #" << n << " 产生（生产者）\n";

    // 如果 m <= n，说明没有跨 if 计算块的依赖（消费者在生产者之前或同时），跳过
    if (m <= n) {
      llvm::outs() << "[calculateFactor] m <= n，没有跨 if 计算块依赖，跳过\n";
      continue;
    }

    // 计算 requiredBuffers = m - n + 1
    int requiredBuffers = m - n + 1;

    llvm::outs() << "[calculateFactor] 依赖距离: requiredBuffers = " << requiredBuffers
                 << ", x = " << x
                 << ", factor = " << requiredBuffers << "/" << x << "\n";

    // 比较当前 factor 与最大 factor（使用整数比较避免浮点数）
    // 比较 requiredBuffers/x 与 maxRequiredBuffers/maxX
    // 等价于比较 requiredBuffers * maxX 与 maxRequiredBuffers * x
    if (requiredBuffers * maxX > maxRequiredBuffers * x) {
      maxRequiredBuffers = requiredBuffers;
      maxX = x;
      llvm::outs() << "[calculateFactor] 更新最大 factor: (" << maxRequiredBuffers << ", " << maxX << ")\n";
    }
  }

  llvm::outs() << "[calculateFactor] 最终返回: (" << maxRequiredBuffers << ", " << maxX << ")\n";
  return {maxRequiredBuffers, maxX};
}

// 修改 for 循环的迭代次数
// 新公式：newIterCount = ceil(iterCount * requiredBuffers / x) + ifCount
// 新上界：newUb = lb + step * newIterCount
// 使用整数运算，避免浮点数类型转换
scf::ForOp extendForOpIterationCount(
    scf::ForOp oldForOp,
    int ifCount,
    int requiredBuffers,
    int x,
    IRMapping &mapper,
    SmallVector<scf::IfOp> &ifOpsInThisFor)
{
  OpBuilder builder(oldForOp);
  Location loc = oldForOp.getLoc();

  Value originalLowerBound = oldForOp.getLowerBound();
  Value originalUpperBound = oldForOp.getUpperBound();
  Value originalStep = oldForOp.getStep();
  Type ubType = originalStep.getType();

  // 创建 ifCount 常量（整数）
  Value ifCountValue;
  if (ubType.isIndex()) {
    ifCountValue = builder.create<arith::ConstantIndexOp>(loc, ifCount);
  } else if (auto intType = dyn_cast<IntegerType>(ubType)) {
    ifCountValue = builder.create<arith::ConstantIntOp>(loc, ifCount, intType);
  } else {
    auto indexVal = builder.create<arith::ConstantIndexOp>(loc, ifCount);
    ifCountValue = builder.create<arith::IndexCastOp>(loc, ubType, indexVal);
  }

  // 创建 requiredBuffers 常量（整数）
  Value requiredBuffersValue;
  if (ubType.isIndex()) {
    requiredBuffersValue = builder.create<arith::ConstantIndexOp>(loc, requiredBuffers);
  } else if (auto intType = dyn_cast<IntegerType>(ubType)) {
    requiredBuffersValue = builder.create<arith::ConstantIntOp>(loc, requiredBuffers, intType);
  } else {
    auto indexVal = builder.create<arith::ConstantIndexOp>(loc, requiredBuffers);
    requiredBuffersValue = builder.create<arith::IndexCastOp>(loc, ubType, indexVal);
  }

  // 创建 x 常量（整数）
  Value xValue;
  if (ubType.isIndex()) {
    xValue = builder.create<arith::ConstantIndexOp>(loc, x);
  } else if (auto intType = dyn_cast<IntegerType>(ubType)) {
    xValue = builder.create<arith::ConstantIntOp>(loc, x, intType);
  } else {
    auto indexVal = builder.create<arith::ConstantIndexOp>(loc, x);
    xValue = builder.create<arith::IndexCastOp>(loc, ubType, indexVal);
  }

  // 计算迭代次数扩展后的上界
  // 原始迭代次数 iterCount = ceil((ub - lb) / step)
  // 新迭代次数 newIterCount = ceil(iterCount * requiredBuffers / x) + ifCount
  // 新上界 newUb = lb + step * newIterCount

  // 1. 计算 range = ub - lb
  Value rangeDiff = builder.create<arith::SubIOp>(loc, originalUpperBound, originalLowerBound);

  // 2. 计算 iterCount = ceil(range / step)（整数）
  Value iterCount;
  if (ubType.isIndex()) {
    iterCount = builder.create<arith::CeilDivUIOp>(loc, rangeDiff, originalStep);
  } else if (auto intType = dyn_cast<IntegerType>(ubType)) {
    if (intType.isSigned()) {
      iterCount = builder.create<arith::CeilDivSIOp>(loc, rangeDiff, originalStep);
    } else {
      iterCount = builder.create<arith::CeilDivUIOp>(loc, rangeDiff, originalStep);
    }
  } else {
    iterCount = builder.create<arith::CeilDivUIOp>(loc, rangeDiff, originalStep);
  }

  // 3. 计算 iterCount * requiredBuffers（整数乘法）
  Value scaledIterCount = builder.create<arith::MulIOp>(loc, iterCount, requiredBuffersValue);

  // 4. 计算 ceil(scaledIterCount / x)（整数 ceildiv）
  Value ceiledScaledIterCount;
  if (ubType.isIndex()) {
    ceiledScaledIterCount = builder.create<arith::CeilDivUIOp>(loc, scaledIterCount, xValue);
  } else if (auto intType = dyn_cast<IntegerType>(ubType)) {
    if (intType.isSigned()) {
      ceiledScaledIterCount = builder.create<arith::CeilDivSIOp>(loc, scaledIterCount, xValue);
    } else {
      ceiledScaledIterCount = builder.create<arith::CeilDivUIOp>(loc, scaledIterCount, xValue);
    }
  } else {
    ceiledScaledIterCount = builder.create<arith::CeilDivUIOp>(loc, scaledIterCount, xValue);
  }

  // 5. 计算 newIterCount = ceiledScaledIterCount + ifCount（整数加法）
  Value newIterCount = builder.create<arith::AddIOp>(loc, ceiledScaledIterCount, ifCountValue);

  // 6. 计算 step * newIterCount
  Value totalSteps = builder.create<arith::MulIOp>(loc, originalStep, newIterCount);

  // 7. 计算 newUpperBound = lb + step * newIterCount
  Value newUpperBound = builder.create<arith::AddIOp>(loc, originalLowerBound, totalSteps);

  SmallVector<Value> newInitArgs(oldForOp.getInitArgs().begin(),
                                 oldForOp.getInitArgs().end());

  auto newForOp = builder.create<scf::ForOp>(
      loc,
      originalLowerBound,
      newUpperBound,
      originalStep,
      newInitArgs);

  for (auto &attr : oldForOp->getAttrs()) {
    newForOp->setAttr(attr.getName(), attr.getValue());
  }

  mapper.map(oldForOp.getInductionVar(), newForOp.getInductionVar());

  for (auto [oldArg, newArg] :
       llvm::zip(oldForOp.getRegionIterArgs(),
                 newForOp.getRegionIterArgs())) {
    mapper.map(oldArg, newArg);
  }

  Block *oldBlock = oldForOp.getBody();
  Block *newBlock = newForOp.getBody();

  unsigned totalArgs = oldBlock->getNumArguments();
  for (unsigned i = 0; i < totalArgs; ++i) {
    BlockArgument oldArg = oldBlock->getArgument(i);
    BlockArgument newArg = newBlock->getArgument(i);
    oldArg.replaceAllUsesWith(newArg);
  }

  Operation *oldTerminator = oldBlock->getTerminator();

  builder.setInsertionPointToStart(newBlock);
  for (Operation &op : llvm::make_early_inc_range(oldBlock->without_terminator())) {
    builder.clone(op, mapper);
  }

  auto oldYield = cast<scf::YieldOp>(oldTerminator);
  SmallVector<Value> newYieldOperands;
  for (unsigned i = 0; i < oldYield.getNumOperands(); ++i) {
    newYieldOperands.push_back(mapper.lookupOrDefault(oldYield.getOperand(i)));
  }

  builder.setInsertionPointToEnd(newBlock);
  builder.create<scf::YieldOp>(loc, newYieldOperands);
  oldYield.erase();

  unsigned numOriginalResults = oldForOp.getNumResults();
  if (numOriginalResults > 0) {
    SmallVector<Value> originalResults;
    for (unsigned i = 0; i < numOriginalResults; ++i) {
      originalResults.push_back(newForOp.getResult(i));
    }
    oldForOp.replaceAllUsesWith(originalResults);
  }

  return newForOp;
}

// 替换 ifOp 中使用 for 循环计数器的操作数为计算块对应的计数器
void replaceForOpCounterInIfOps(
    SmallVector<scf::IfOp> ifOpsInThisFor,
    scf::ForOp oldForOp,
    scf::ForOp newForOp,
    IRMapping &mapper,
    DenseMap<scf::IfOp, Value> &cntArgs) {

  for (scf::IfOp oldIfOp : ifOpsInThisFor) {
    scf::IfOp newIfOp = dyn_cast<scf::IfOp>(mapper.lookupOrDefault(oldIfOp));
    Value newCntVal = mapper.lookupOrDefault(cntArgs[oldIfOp]);
    Value oldIndVar = oldForOp.getInductionVar();
    Value newIndVar = mapper.lookupOrDefault(oldIndVar);

    newIfOp.walk([&](Operation *op) {
      for (OpOperand &operand : op->getOpOperands()) {
        if (operand.get() == newIndVar) {
          operand.set(newCntVal);
        }
      }
    });

    cntArgs.erase(oldIfOp);
    cntArgs[newIfOp] = newCntVal;
  }
}

// ========== 临时逻辑：更新 intraCoreDependentMap 的 value ==========
// 对于 intraCoreDependentMap 中每个条目，将 value 更新为包含 key 的第一个 operand 的 SmallVector
void updateIntraCoreDependentMapTemp(
    DenseMap<scf::ForOp, DenseMap<Value, SmallVector<Value>>> &intraCoreDependentMap) {
  llvm::outs() << "=== [临时逻辑] 更新 intraCoreDependentMap ===\n";
  for (auto &forOpEntry : intraCoreDependentMap) {
    llvm::outs() << "  ForOp: " << forOpEntry.first << "\n";
    for (auto &innerEntry : forOpEntry.second) {
      Value key = innerEntry.first;
      SmallVector<Value> &valueList = innerEntry.second;

      // 获取定义这个 key 的 op
      if (auto *defOp = key.getDefiningOp()) {
        // 获取第一个 operand
        if (defOp->getNumOperands() > 0) {
          Value firstOperand = defOp->getOperand(0);
          // 清空原有列表并添加第一个 operand
          valueList.clear();
          valueList.push_back(firstOperand);
          llvm::outs() << "    [临时逻辑] 更新: key=" << key
                       << ", 新 value=[" << firstOperand << "]\n";
        } else {
          llvm::outs() << "    [临时逻辑] key=" << key << " 的定义 op 没有 operand，跳过\n";
        }
      } else {
        llvm::outs() << "    [临时逻辑] key=" << key << " 不是由 op 定义，跳过\n";
      }
    }
  }
  llvm::outs() << "=== [临时逻辑] 更新完成 ===\n";
}
// ========== 临时逻辑结束 ==========

void AddIfControlsPass::UpdateForIterTimes(ModuleOp module)
{
  // 打印 cntArgs
  llvm::outs() << "=== cntArgs ===\n";
  llvm::outs() << "cntArgs.size() = " << cntArgs.size() << "\n";
  for (auto &[ifOp, cntVal] : cntArgs) {
    if (auto* defOp = cntVal.getDefiningOp()) {
      llvm::outs() << "  ifOp cond: " << ifOp.getCondition()
                   << " -> found: " << defOp->getName()
                   << " -> " << cntVal << "\n";
    } else {
      if (auto arg = mlir::dyn_cast<BlockArgument>(cntVal)) {
        llvm::outs() << "  ifOp cond: " << ifOp.getCondition()
                     << " -> found blockArg: " << cntVal
                     << " (blockArg #" << arg.getArgNumber() << " of ";
        arg.getOwner()->print(llvm::outs());
        llvm::outs() << ")\n";
      } else {
        llvm::outs() << "  ifOp cond: " << ifOp.getCondition()
                     << " -> found: " << cntVal << "\n";
      }
    }
  }
  llvm::outs().flush();

  // 按 ssbuffer.main_loop 属性值分组 for 循环
  // CV 两边的 for 循环具有相同的 main_loop id，需要一起修改
  DenseMap<int, SmallVector<scf::ForOp>> forOpsByMainLoopId;
  module.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr("ssbuffer.main_loop")) {
      auto mainLoopId = forOp->getAttrOfType<IntegerAttr>("ssbuffer.main_loop");
      if (mainLoopId) {
        int id = mainLoopId.getInt();
        forOpsByMainLoopId[id].push_back(forOp);
        llvm::outs() << "[分组] forOp: " << forOp << " -> main_loop id = " << id << "\n";
      }
    }
  });

  llvm::outs() << "=== forOpsByMainLoopId 分组结果 ===\n";
  for (auto &entry : forOpsByMainLoopId) {
    llvm::outs() << "  main_loop id = " << entry.first << ": " << entry.second.size() << " 个 for 循环\n";
    for (auto forOp : entry.second) {
      llvm::outs() << "    " << forOp << "\n";
    }
  }

  // 对每组 for 循环进行处理
  for (auto &entry : forOpsByMainLoopId) {
    int mainLoopId = entry.first;
    SmallVector<scf::ForOp> &forOpsInGroup = entry.second;

    llvm::outs() << "\n=== 处理 main_loop id = " << mainLoopId << " 的 for 循环组 ===\n";

    // 计算组内所有 for 循环的最大 ifCount 和最大 factor
    int maxIfCount = 0;
    int maxRequiredBuffers = 1;
    int maxX = 1;

    // 存储每个 for 循环的 ifOps 信息
    DenseMap<scf::ForOp, SmallVector<scf::IfOp>> ifOpsByForOp;

    for (scf::ForOp forOp : forOpsInGroup) {
      llvm::outs() << "  计算 forOp: " << forOp << " 的参数\n";

      int ifCount = 0;
      SmallVector<scf::IfOp> ifOpsInThisFor;
      for (auto &[ifOp, cntVal] : cntArgs) {
        if (ifOp->hasAttr("ssbuffer.if") && ifOp->getParentOp() == forOp) {
          ifCount++;
          ifOpsInThisFor.push_back(ifOp);
        }
      }
      llvm::outs() << "    ifCount: " << ifCount << "\n";

      ifOpsByForOp[forOp] = ifOpsInThisFor;

      // 更新最大 ifCount
      if (ifCount > maxIfCount) {
        maxIfCount = ifCount;
      }

      // 计算 factor 并更新最大值
      auto [requiredBuffers, x] = calculateFactor(forOp, intraCoreDependentMap);
      llvm::outs() << "    calculated: requiredBuffers = " << requiredBuffers
                   << ", x = " << x << "\n";

      // 使用整数比较来更新最大 factor
      // 比较 requiredBuffers/x 与 maxRequiredBuffers/maxX
      // 等价于比较 requiredBuffers * maxX 与 maxRequiredBuffers * x
      if (requiredBuffers * maxX > maxRequiredBuffers * x) {
        maxRequiredBuffers = requiredBuffers;
        maxX = x;
        llvm::outs() << "    更新最大 factor: (" << maxRequiredBuffers << ", " << maxX << ")\n";
      }
    }

    llvm::outs() << "  组内最大值: maxIfCount = " << maxIfCount
                 << ", maxRequiredBuffers = " << maxRequiredBuffers
                 << ", maxX = " << maxX << "\n";

    // 如果 maxIfCount 为 0，跳过该组
    if (maxIfCount == 0) {
      llvm::outs() << "  maxIfCount = 0，跳过该组\n";
      continue;
    }

    // 对组内所有 for 循环使用相同的最大值进行修改
    for (scf::ForOp oldForOp : forOpsInGroup) {
      llvm::outs() << "  修改 forOp: " << oldForOp << "\n";

      SmallVector<scf::IfOp> &ifOpsInThisFor = ifOpsByForOp[oldForOp];

      IRMapping mapper;
      scf::ForOp newForOp = extendForOpIterationCount(oldForOp, maxIfCount, maxRequiredBuffers, maxX, mapper, ifOpsInThisFor);

      replaceForOpCounterInIfOps(ifOpsInThisFor, oldForOp, newForOp, mapper, cntArgs);

      oldForOp.erase();
    }

    llvm::outs() << "  组内所有 for 循环已修改完成\n";
  }
}

void AddIfControlsPass::initDependentMap(ModuleOp module) {
  // 从全局单例 BufferRelationAnalysis 获取数据到 intraCoreDependentMap
  auto &analysis = mlir::triton::getGlobalBufferRelation();
  const auto &bufferMap = analysis.getAllBufferRelations();

  llvm::outs() << "=== initDependentMap: 从全局单例迁移 buffer 数据 ===\n";
  llvm::outs() << "BufferRelationAnalysis 大小: " << bufferMap.size() << "\n";

  for (auto &entry : bufferMap) {
    scf::ForOp forOp = entry.first;
    auto &innerMap = entry.second;

    llvm::outs() << "处理 forOp: " << forOp << "\n";
    llvm::outs() << "  selectedBuffer 数量: " << innerMap.size() << "\n";

    DenseMap<Value, SmallVector<Value>> localInnerMap;
    for (auto &innerEntry : innerMap) {
      Value selectedBuffer = innerEntry.first;
      SmallVector<Value> memrefs = innerEntry.second;
      llvm::outs() << "  selectedBuffer: " << selectedBuffer << "\n";
      llvm::outs() << "    memrefs 数量: " << memrefs.size() << "\n";
      for (auto v : memrefs) {
        llvm::outs() << "      " << v << "\n";
      }

      localInnerMap[selectedBuffer] = memrefs;
    }

    intraCoreDependentMap[forOp] = localInnerMap;
  }

  // 从 BufferRelationAnalysis 获取 crossCore 依赖数据
  const auto &crossCoreRelations = analysis.getCrossCoreRelations();

  llvm::outs() << "=== initDependentMap: 迁移 crossCore 数据 ===\n";
  llvm::outs() << "crossCoreRelations 大小: " << crossCoreRelations.size() << "\n";

  for (auto &entry : crossCoreRelations) {
    Value consumer = entry.first;
    SmallVector<Value> producers = entry.second;

    llvm::outs() << "  consumer: " << consumer << "\n";
    llvm::outs() << "    producers 数量: " << producers.size() << "\n";
    for (auto v : producers) {
      llvm::outs() << "      " << v << "\n";
    }
    crossCoreDependentMap[consumer] = producers;
  }

  // ========== 临时逻辑：为 crossCoreDependentMap 设置映射 ==========
  // 查找带有 ssbuffer.main_loop 属性的 for 循环
  // 在这些 for 循环中查找 hivm.hir.convert_layout 和 memref.memory_space_cast op
  // 设置 crossCoreDependentMap: key = op.getResult(0), value = {op.getOperand(0)}
  // 要求 operand 的 definingOp 必须是 memref::AllocOp

  module.walk([&](scf::ForOp forOp) {
    // 检查是否有 ssbuffer.main_loop 属性
    if (!forOp->hasAttr("ssbuffer.main_loop")) {
      return;
    }

    llvm::outs() << "[临时逻辑] 找到 main_loop for 循环: " << forOp << "\n";

    // 在该 for 循环内查找 memref::MemorySpaceCastOp
    forOp.walk([&](memref::MemorySpaceCastOp memspacecastOp) {
      Value result = memspacecastOp.getResult();
      Value operand = memspacecastOp.getOperand();

      // 检查 operand 的 definingOp 是否是 memref::AllocOp
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || !isa<memref::AllocOp>(defOp)) {
        llvm::outs() << "[临时逻辑] memspacecast op 的 operand 不是 AllocOp 的 result，跳过\n";
        llvm::outs() << "  operand: " << operand << "\n";
        llvm::outs() << "  definingOp: " << (defOp ? defOp->getName().getStringRef() : "null") << "\n";
        return;
      }

      llvm::outs() << "[临时逻辑] 找到 memspacecast op (operand 是 AllocOp result)\n";
      llvm::outs() << "  key (result): " << result << "\n";
      llvm::outs() << "  value (operand): " << operand << "\n";

      // key = result, value = {operand}
      crossCoreDependentMap[result] = {operand};
    });

    // 在该 for 循环内查找 hivm::ConvertLayoutOp
    forOp.walk([&](hivm::ConvertLayoutOp convertOp) {
      Value result = convertOp.getResult();
      Value operand = convertOp.getOperand(0);

      // 检查 operand 的 definingOp 是否是 memref::AllocOp
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || !isa<memref::AllocOp>(defOp)) {
        llvm::outs() << "[临时逻辑] convert_layout op 的 operand 不是 AllocOp 的 result，跳过\n";
        llvm::outs() << "  operand: " << operand << "\n";
        llvm::outs() << "  definingOp: " << (defOp ? defOp->getName().getStringRef() : "null") << "\n";
        return;
      }

      llvm::outs() << "[临时逻辑] 找到 convert_layout op (operand 是 AllocOp result)\n";
      llvm::outs() << "  key (result): " << result << "\n";
      llvm::outs() << "  value (operand): " << operand << "\n";

      // key = result, value = {operand}
      crossCoreDependentMap[result] = {operand};
    });
  });

  // 打印临时的 crossCoreDependentMap 内容
  llvm::outs() << "=== [临时逻辑] crossCoreDependentMap 内容 ===\n";
  for (auto &entry : crossCoreDependentMap) {
    llvm::outs() << "  key: " << entry.first << "\n";
    llvm::outs() << "    value (producers): ";
    for (auto v : entry.second) {
      llvm::outs() << v << ", ";
    }
    llvm::outs() << "\n";
  }
  // ========== 临时逻辑结束 ==========

  llvm::outs() << "=== initDependentMap: 迁移完成 ===\n";
  llvm::outs() << "intraCoreDependentMap 大小: " << intraCoreDependentMap.size() << "\n";
  llvm::outs() << "crossCoreDependentMap 大小: " << crossCoreDependentMap.size() << "\n";
  llvm::outs().flush();
}

// 对于main_loop的forOp(parentOp能找到是hivm::TCoreType::CUBE的scopeOp), 分析不同block_id间op的依赖，并且把这些依赖添加到forOp的iter_args中，创建新的forOp替换原有的forOp
// tensor类型的依赖的iter_args的初始值: arith.constant + tensor.empty + linalg.fill实现
// 标量类型的依赖的iter_args的初始值: arith.constant实现
// index类型的依赖的iter_args的初始值: arith.constant + arith.index_cast实现
// memref类型的依赖的iter_args的初始值: memref.alloc实现
// 并且依据上面的信息更新intraCoreDependentMap
void AddIfControlsPass::updateInputAndInitMap(ModuleOp module) {
  // 第一步：遍历 main_loop forOp，分析依赖，收集到 toProcess
  SmallVector<std::pair<scf::ForOp, DenseMap<Value, SmallVector<Value>>>> toProcess;

  module.walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr("ssbuffer.main_loop"))
      return;
    scope::ScopeOp parentScope = forOp->getParentOfType<scope::ScopeOp>();
    if (!parentScope)
      return;
    auto tcoreType = parentScope->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
    if (!tcoreType)
      return;
    bool isVectorCore = (tcoreType == hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR));
    if (!isVectorCore)
      return;

    // 构建依赖关系
    DenseMap<Value, SmallVector<Value>> localMap;
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (!op.getAttrOfType<IntegerAttr>("ssbuffer.block_id"))
        continue;
      if (op.getNumResults() == 0)
        continue;

      for (OpOperand &operand : op.getOpOperands()) {
        Value operandValue = operand.get();
        if (auto *defOp = operandValue.getDefiningOp()) {
          if (defOp->getBlock() != op.getBlock())
            continue;
          if (defOp->getNumResults() == 0)
            continue;
          if (auto blockIdAttr = defOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
            int operandBlockId = blockIdAttr.getInt();
            int opBlockId = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id").getInt();
            if (operandBlockId != opBlockId) {
              localMap[op.getResult(0)].push_back(defOp->getResult(0));
              llvm::outs() << "[核内依赖] op: " << op << "/defOp: " << *defOp << "有依赖关系\n";
            }
          }
        }
      }
    }

    if (!localMap.empty()) {
      toProcess.push_back({forOp, localMap});
      llvm::outs() << "  VECTOR forOp: " << forOp << " 有 " << localMap.size() << " 个依赖\n";
    }
  });

  // 设置 intraCoreDependentMap
  for (auto &[forOp, localMap] : toProcess) {
    intraCoreDependentMap[forOp] = localMap;
  }

  // 第二步：遍历 toProcess，创建新 forOp，添加 iter_args，更新 yield，erase old
  for (auto &[oldForOp, localMap] : toProcess) {
    OpBuilder builder(oldForOp);
    Location loc = oldForOp.getLoc();

    // 根据依赖类型创建初始值
    SmallVector<Value> initValues;
    for (auto &entry : localMap) {
      Value depValue = entry.second[0];
      Type valueType = depValue.getType();

      if (auto tensorType = dyn_cast<RankedTensorType>(valueType)) {
        // ✅ 自动获取 tensor 元素类型，不写死 f32
        Type elemType = tensorType.getElementType();
        TypedAttr zeroAttr;

        // 浮点类型
        if (isa<FloatType>(elemType)) {
          zeroAttr = builder.getFloatAttr(elemType, 0.0f);
        }
        // 整数类型
        else if (isa<IntegerType>(elemType)) {
          zeroAttr = builder.getIntegerAttr(elemType, 0);
        }
        // index 类型
        else if (isa<IndexType>(elemType)) {
          zeroAttr = builder.getIndexAttr(0);
        }
        // 不支持的类型
        else {
          llvm_unreachable("Unsupported element type for tensor init");
        }

        Value zero = builder.create<arith::ConstantOp>(loc, zeroAttr);
        Value empty = builder.create<tensor::EmptyOp>(loc, tensorType.getShape(), elemType);
        Value fill = builder.create<linalg::FillOp>(loc, zero, empty).getResult(0);
        initValues.push_back(fill);
      }
      // index 类型
      else if (isa<IndexType>(valueType)) {
        auto zeroIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
        initValues.push_back(zeroIdx);
      }
      // memref 类型
      else if (isa<MemRefType>(valueType)) {
        auto allocOp = builder.create<memref::AllocOp>(loc, mlir::cast<MemRefType>(valueType));
        initValues.push_back(allocOp.getResult());
      }
      // 普通整数
      else if (isa<IntegerType>(valueType)) {
        auto constOp = builder.create<arith::ConstantIntOp>(loc, 0, valueType);
        initValues.push_back(constOp);
      }
      // 普通浮点
      else if (isa<FloatType>(valueType)) {
        auto constOp = builder.create<arith::ConstantFloatOp>(loc, APFloat(0.0), mlir::cast<FloatType>(valueType));
        initValues.push_back(constOp);
      }
    }

    if (initValues.empty())
      continue;

    // 收集原始 initArgs + 新增的依赖初始值
    SmallVector<Value> newInitArgs(oldForOp.getInitArgs().begin(), oldForOp.getInitArgs().end());
    newInitArgs.append(initValues.begin(), initValues.end());

    // 创建新的 forOp
    scf::ForOp newForOp = builder.create<scf::ForOp>(
        loc, oldForOp.getLowerBound(), oldForOp.getUpperBound(), oldForOp.getStep(), newInitArgs);

    for (auto &attr : oldForOp->getAttrs())
      newForOp->setAttr(attr.getName(), attr.getValue());

    Block *oldBlock = oldForOp.getBody();
    Block *newBlock = newForOp.getBody();

    // 替换 block arguments
    for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
      oldBlock->getArgument(i).replaceAllUsesWith(newBlock->getArgument(i));
    }

    // 移动操作
    Operation *oldTerminator = oldBlock->getTerminator();
    for (Operation &op : llvm::make_early_inc_range(oldBlock->without_terminator())) {
      op.moveBefore(newBlock, newBlock->end());
    }

    // 构建 yield operands
    SmallVector<Value> yieldOperands;
    if (auto oldYield = dyn_cast<scf::YieldOp>(oldTerminator)) {
      yieldOperands.append(oldYield.getOperands().begin(), oldYield.getOperands().end());
    }
    // 添加依赖的 producer 到 yield
    for (auto &entry : localMap) {
      yieldOperands.append(entry.second.begin(), entry.second.end());
    }

    builder.setInsertionPointToEnd(newBlock);
    builder.create<scf::YieldOp>(loc, yieldOperands);

    if (intraCoreDependentMap.count(oldForOp)) {
      intraCoreDependentMap[newForOp] = intraCoreDependentMap[oldForOp];
      intraCoreDependentMap.erase(oldForOp);
    }

    // 替换并 erase
    if (oldForOp.getNumResults() > 0) {
      SmallVector<Value> newResults;
      for (unsigned i = 0; i < oldForOp.getNumResults(); ++i)
        newResults.push_back(newForOp.getResult(i));
      oldForOp.replaceAllUsesWith(newResults);
    }
    oldTerminator->erase();
    oldForOp.erase();
  }
}

void AddIfControlsPass::runOnOperation() {
  ModuleOp module = getOperation();

  llvm::outs()<<"before addifcontrols:\n";
  llvm::outs()<<module<<"\n\n";
  llvm::outs().flush();

  initDependentMap(module);

  updateInputAndInitMap(module);
  llvm::outs()<<"after updateInputAndInitMap:\n";
  llvm::outs()<<module<<"\n\n";
  llvm::outs().flush();

  //   llvm::outs() << "print intra\n";
  // for (auto pair1 : intraCoreDependentMap) {
  //   scf::ForOp forOp = pair1.first;
  //   llvm::outs() << "key value size: " << pair1.second.size() << "\n";
  // }

  createIfOps(module);
  llvm::outs()<<"after createIfOps:\n";
  llvm::outs()<<module<<"\n\n";
  llvm::outs().flush();

  // // ========== 临时逻辑：填充 intraCoreDependentMap 中空列表的 operand ==========
  // // 对于 intraCoreDependentMap 中 value 为空列表的条目，将 key 对应的 op 的第一个 operand push 到列表中
  // for (auto &forOpEntry : intraCoreDependentMap) {
  //   for (auto &innerEntry : forOpEntry.second) {
  //     Value key = innerEntry.first;
  //     SmallVector<Value> &valueList = innerEntry.second;

  //     // 如果列表为空，需要填充
  //     if (valueList.empty()) {
  //       // 获取定义这个 key 的 op
  //       if (auto *defOp = key.getDefiningOp()) {
  //         // 获取第一个 operand
  //         if (defOp->getNumOperands() > 0) {
  //           Value firstOperand = defOp->getOperand(0);
  //           valueList.push_back(firstOperand);
  //           llvm::outs() << "[临时逻辑] 填充 intraCoreDependentMap: key=" << key
  //                        << ", operand=" << firstOperand << "\n";
  //         }
  //       }
  //     }
  //   }
  // }
  // // ========== 临时逻辑结束 ==========

  // TODO: InnerDepNums from upstream pass
  updateForOps(module);
  llvm::outs()<<"after updateForOps:\n";
  llvm::outs()<<module<<"\n\n";
  llvm::outs().flush();


  llvm::outs()<<"blockCounters: \n";
  for (auto &entry : blockCounters) {
    llvm::outs() << "ForOp @ " << entry.first << " :: ";
    llvm::outs() << "\n";
    for (auto v : entry.second) {
      llvm::outs()<< v << ", ";
    }
    llvm::outs() << "\n\n";
  }

  llvm::outs()<<"innerDepConds: \n";
  for (auto &entry : innerDepConds) {
    llvm::outs() << "ForOp @ " << entry.first << " :: ";
    llvm::outs() << "\n";
    for (auto v : entry.second) {
      llvm::outs()<< v << ", ";
    }
    llvm::outs() << "\n\n";
  }

  llvm::outs() << "intraCoreDependentMap: \n";
  for (auto &entry : intraCoreDependentMap) {
    llvm::outs() << "ForOp @ " << entry.first << " :: ";
    llvm::outs() << "\n";
    for (auto &innerEntry : entry.second) {
      llvm::outs() << "  selectedDep: " << innerEntry.first << " -> outputs: [";
      for (auto v : innerEntry.second) {
        llvm::outs() << v << ", ";
      }
      llvm::outs() << "]\n";
    }
    llvm::outs() << "\n";
  }

  // // 打印 intraCoreDependentMap（updateIfConds 前）
  // llvm::outs() << "=== intraCoreDependentMap (before updateIfConds) ===\n";
  // llvm::outs() << "intraCoreDependentMap.size() = " << intraCoreDependentMap.size() << "\n";
  // for (auto &forOpEntry : intraCoreDependentMap) {
  //   llvm::outs() << "  ForOp: " << forOpEntry.first << "\n";
  //   for (auto &innerEntry : forOpEntry.second) {
  //     llvm::outs() << "    key (consumerResult): " << innerEntry.first << "\n";
  //     llvm::outs() << "    value (producerBuffers): [";
  //     for (auto v : innerEntry.second) {
  //       llvm::outs() << v << ", ";
  //     }
  //     llvm::outs() << "]\n";
  //   }
  // }
  // llvm::outs().flush();

  updateIfConds(module);
  llvm::outs()<<"after updateIfConds:\n";
  llvm::outs()<<module<<"\n\n";

  // // ========== 临时逻辑：更新 intraCoreDependentMap ==========
  // updateIntraCoreDependentMapTemp(intraCoreDependentMap);

  // 打印 intraCoreDependentMap（临时更新后）
  llvm::outs() << "=== intraCoreDependentMap (after temp update) ===\n";
  llvm::outs() << "intraCoreDependentMap.size() = " << intraCoreDependentMap.size() << "\n";
  for (auto &forOpEntry : intraCoreDependentMap) {
    llvm::outs() << "  ForOp: " << forOpEntry.first << "\n";
    for (auto &innerEntry : forOpEntry.second) {
      llvm::outs() << "    key (consumerResult): " << innerEntry.first << "\n";
      llvm::outs() << "    value (producerBuffers): [";
      for (auto v : innerEntry.second) {
        llvm::outs() << v << ", ";
      }
      llvm::outs() << "]\n";
    }
  }
  llvm::outs().flush();
  // // ========== 临时逻辑结束 ==========

  // constructCntArgs(module, cntArgs);

  UpdateForIterTimes(module);
  llvm::outs()<<"after UpdateForIterTimes:\n";
  llvm::outs()<<module<<"\n\n";

  return ;
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createAddIfControlsPass() {
  return std::make_unique<AddIfControlsPass>();
}