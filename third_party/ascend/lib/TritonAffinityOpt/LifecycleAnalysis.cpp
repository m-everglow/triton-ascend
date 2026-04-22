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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_LIFECYCLEANALYSIS
#include "ascend/include/TritonAffinityOpt/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {
struct LifecycleAnalysisPass
    : public mlir::triton::impl::LifecycleAnalysisBase<
          LifecycleAnalysisPass> {
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<linalg::LinalgDialect>();
  }
  void setDepthRecursive(Operation *op, int depth, MLIRContext *ctx);
  void getNestingDepth(ModuleOp module);
  void getEndNestingDepth(ModuleOp module);
  void getEndNestingDepthRecursive(Operation *op, MLIRContext *ctx);
  int32_t getEndNestingDepthForValue(Value value, MLIRContext *ctx);

};
} // namespace

void LifecycleAnalysisPass::setDepthRecursive(Operation *op, int depth, MLIRContext *ctx) {
  op->setAttr(
      "ssbuffer.nesting_depth",
      IntegerAttr::get(IntegerType::get(ctx, 32), depth));

  int nextDepth = depth;

  if (isa<scf::ForOp>(op) ||
      isa<scf::IfOp>(op) ||
      isa<scope::ScopeOp>(op)) {
    nextDepth = depth + 1;
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &innerOp : block.getOperations()) {
        setDepthRecursive(&innerOp, nextDepth, ctx);
      }
    }
  }
}

int32_t LifecycleAnalysisPass::getEndNestingDepthForValue(Value value, MLIRContext *ctx) {
    int32_t endDepth = -1;

    for (OpOperand &use : value.getUses()) {
        if (auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner())) {
            auto parentOp = yieldOp->getParentOp();
            auto parentDepthAttr = parentOp->getAttrOfType<IntegerAttr>("ssbuffer.nesting_depth");
            if (parentDepthAttr) {
                int32_t parentDepth = parentDepthAttr.getInt();
                if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
                    for (OpResult result : forOp->getResults()) {
                        int32_t outerEndDepth = getEndNestingDepthForValue(result, ctx);
                        if (outerEndDepth >= 0) {
                            endDepth = (endDepth < 0) ? outerEndDepth : std::min(endDepth, outerEndDepth);
                        }
                    }
                }

                if (endDepth < 0) {
                    endDepth = parentDepth;
                }
            }
            break;
        }
    }
    return endDepth;
}

void LifecycleAnalysisPass::getEndNestingDepthRecursive(Operation *op, MLIRContext *ctx) {
    auto nestingDepthAttr = op->getAttrOfType<IntegerAttr>("ssbuffer.nesting_depth");
    if (nestingDepthAttr) {
        int32_t nestingDepth = nestingDepthAttr.getInt();
        int32_t endNestingDepth = nestingDepth;

        int32_t yieldEndDepth = -1;
        for (OpResult result : op->getResults()) {
            int32_t depth = getEndNestingDepthForValue(result, ctx);
            if (depth >= 0) {
                yieldEndDepth = (yieldEndDepth < 0) ? depth : std::min(yieldEndDepth, depth);
            }
        }

        if (yieldEndDepth >= 0) {
            endNestingDepth = yieldEndDepth;
        }

        op->setAttr(
            "ssbuffer.end_nesting_depth",
            IntegerAttr::get(IntegerType::get(ctx, 32), endNestingDepth));
    }

    for (Region &region : op->getRegions()) {
        for (Block &block : region.getBlocks()) {
            for (Operation &innerOp : block.getOperations()) {
                getEndNestingDepthRecursive(&innerOp, ctx);
            }
        }
    }
}

void LifecycleAnalysisPass::getNestingDepth(ModuleOp module) {
  module.walk([&](triton::FuncOp funcOp) {

    auto *ctx = funcOp.getContext();

    for (Region &region : funcOp->getRegions()) {
      for (Block &block : region.getBlocks()) {
        for (Operation &op : block.getOperations()) {
          setDepthRecursive(&op, /*depth=*/0, ctx);
        }
      }
    }

  });
}

void LifecycleAnalysisPass::getEndNestingDepth(ModuleOp module) {
    module.walk([&](triton::FuncOp funcOp) {
        auto *ctx = funcOp.getContext();

        for (Region &region : funcOp->getRegions()) {
            for (Block &block : region.getBlocks()) {
                for (Operation &op : block.getOperations()) {
                    getEndNestingDepthRecursive(&op, ctx);
                }
            }
        }
    });
}

void LifecycleAnalysisPass::runOnOperation() {
  ModuleOp module = getOperation();

  getNestingDepth(module);

  getEndNestingDepth(module);

  return ;
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createLifecycleAnalysisPass() {
  return std::make_unique<LifecycleAnalysisPass>();
}
