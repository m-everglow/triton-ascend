add_triton_library(TritonAffinityOpt
  DAGSSBuffer.cpp
  DAG.cpp
  DAGSync.cpp
  DAGScope.cpp
  AddIfControls.cpp
  LifecycleAnalysis.cpp
  InnerMultibuffer.cpp
  OuterMultiBuffer.cpp
  TritonSyncScope/Common/FlagIdManager.cpp

  DEPENDS
  TritonAffinityOptConversionPassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  BiShengIRScopeDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
  MLIRSCFDialect
)