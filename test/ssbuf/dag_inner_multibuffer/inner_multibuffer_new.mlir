// InnerMultibuffer UT — §6.2 核内双缓冲（新版，跨 id 边界清晰）
//
// 问题背景：原 UT 中 id 内部有链式依赖，collectBlockInfo 会把所有中间值
// 都当成 depVal，导致过多 buffer。
//
// 新 UT 结构：每个跨 id 依赖都是"一跳"，没有中间链式：
//   id=0: cst_v（仅此一个 op，边界点）
//   id=1: cst_c → load_c → dot_c（仅此一个 op，边界点）
//   id=2: v5（consumer of cst_v）、v8（consumer of dot_c）、v9（合并）
//
// 依赖关系：
//   cst_v(id=0) ──跨 id──→ v5(id=2)  ← 需要 buffer（cst_v 是 producer）
//   dot_c(id=1) ──跨 id──→ v8(id=2)  ← 需要 buffer（dot_c 是 producer）
//
// id=1 内还有 cst_c 和 load_c，但它们是 dot_c 的上游，不跨 id，不需要 buffer。
//
// 预期变换：只需要 2 对 buffer（cst_v 和 dot_c 各一对）
//
// RUN: triton-adapter-opt %s --inner-multibuffer 2>&1 | FileCheck %s

module {
  tt.func @inner_multibuffer_test(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<tensor<64x64xf32>>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant {ssbuffer.nesting_depth = 0 : i32} 0 : index
    %c64 = arith.constant {ssbuffer.nesting_depth = 0 : i32} 64 : index
    %c1024 = arith.constant {ssbuffer.nesting_depth = 0 : i32} 1024 : index
    %cst_init = arith.constant {ssbuffer.nesting_depth = 0 : i32} dense<0.000000e+00> : tensor<64x64xf32>

    scope.scope : () -> () {
      %loop = scf.for %iv = %c0 to %c1024 step %c64 iter_args(%acc = %cst_init) -> tensor<64x64xf32> {

        // ===== id=0: 仅一个 op，作为跨 id 的 producer =====
        // %cst_v: 跨 id producer，写入 buffer（被 id=2 的 v5 依赖）
        %cst_v = arith.constant {ssbuffer.core_type = "vector", ssbuffer.id = 0 : i32, ssbuffer.nesting_depth = 2 : i32} dense<1.000000e+00> : tensor<64x64xf32>

        // ===== id=1: producer chain（不跨 id），dot_c 作为跨 id producer =====
        // %cst_c: id=1 内部 producer，不跨 id，不需要 buffer
        %cst_c = arith.constant {ssbuffer.core_type = "cube", ssbuffer.id = 1 : i32, ssbuffer.nesting_depth = 2 : i32} dense<0.000000e+00> : tensor<64x64xf32>
        // %load_c: id=1 内部 consumer，不跨 id，不需要 buffer
        %load_c = tt.load %arg1 {ssbuffer.core_type = "cube", ssbuffer.id = 1 : i32, ssbuffer.nesting_depth = 2 : i32} : !tt.ptr<tensor<64x64xf32>>
        // %dot_c: 跨 id producer，写入 buffer（被 id=2 的 v8 依赖）
        %dot_c = tt.dot %cst_v, %load_c, %cst_c {ssbuffer.core_type = "cube", ssbuffer.id = 1 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>

        // ===== id=2: consumers =====
        // %v5: consumer of %cst_v(id=0)，从 buffer 读取
        %v5 = arith.addf %cst_v, %cst_v {ssbuffer.core_type = "vector", ssbuffer.id = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64x64xf32>
        // %v8: consumer of %dot_c(id=1)，从 buffer 读取
        %v8 = arith.addf %dot_c, %dot_c {ssbuffer.core_type = "vector", ssbuffer.id = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64x64xf32>

        // ===== id=2: 合并计算 =====
        %v9 = arith.addf %v5, %v8 {ssbuffer.core_type = "vector", ssbuffer.id = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64x64xf32>

        scf.yield %v9 : tensor<64x64xf32>
      } {ssbuffer.mainloop, ssbuffer.nesting_depth = 2 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, ssbuffer.nesting_depth = 1 : i32}

    tt.return
  }
}
