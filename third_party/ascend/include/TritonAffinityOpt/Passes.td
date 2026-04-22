/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_AFFINITY_OPTIMIZATION_PASSES
#define TRITON_AFFINITY_OPTIMIZATION_PASSES

include "mlir/Pass/PassBase.td"

def DAGSSBuffer : Pass<"dag-ssbuf", "mlir::ModuleOp"> {
  let summary = "Convert vector operations to shared storage buffer operations";
  let constructor = "triton::createDAGSSBufferPass()";
  let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def DAGScope : Pass<"dag-scope", "mlir::ModuleOp"> {
  let summary = "Convert native triton code to NPU-affine code";
  let constructor = "triton::createDAGScopePass()";
  let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def DAGSync : Pass<"dag-sync", "mlir::ModuleOp"> {
  let summary = "DAG sync";
  let constructor = "triton::createDAGSyncPass()";
  let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "annotation::AnnotationDialect"];
}

def AddIfControls : Pass<"add-if-controls", "mlir::ModuleOp"> {
  let summary = "add if controls for ssbuffer";
  let constructor = "triton::createAddIfControlsPass()";
  let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def LifecycleAnalysis : Pass<"lifecycle-analysis", "mlir::ModuleOp"> {
  let summary = "Analyze operation lifecycles";
  let constructor = "triton::createLifecycleAnalysisPass()";
  let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def MultiBuffer : Pass<"inner-multibuffer", "mlir::ModuleOp"> {
  let summary = "Intra-scope multi-buffer allocation for SSBUF";
  let constructor = "triton::createMultiBufferPass()";
  let dependentDialects = [
    "hivm::HIVMDialect",
    "bufferization::BufferizationDialect",
    "scope::ScopeDialect",
    "LLVM::LLVMDialect"
  ];
}
def OuterMultiBuffer : Pass<"outer-multibuffer", "mlir::ModuleOp"> {
  let summary = "Inter-scope (CV) multi-buffer allocation for SSBUF";
  let constructor = "triton::createOuterMultiBufferPass()";
  let dependentDialects = [
    "hivm::HIVMDialect",
    "scope::ScopeDialect",
    "LLVM::LLVMDialect"
  ];
}

#endif // TRITON_AFFINITY_OPTIMIZATION_PASSES