// InnerMultibuffer UT — §6.2 核内双缓冲
//
// 核心约束：只处理带 ssbuffer.mainloop 标签的 scf.for 循环
//
// 依赖关系（非链式，直接跨 id 依赖）：
//   Group 1: %v1(id=0) → %v3(id=1)
//     - %v1 是生产者（被 id=1 的 %v3 依赖）
//     - %v3 是消费者（依赖 id=0 的 %v1）
//     → 需要在 %v1 和 %v3 之间插入双缓冲
//
//   Group 2: %v2(id=0) → %v5(id=2) → %v6(id=2)
//     - %v2 是生产者（被 id=2 的 %v5 依赖）
//     - %v5 是消费者（依赖 id=0 的 %v2）
//     - %v6 是 v5 的下游（确保 v5 在 depUserMap 中是端点）
//     → 需要在 %v2 和 %v5 之间插入双缓冲
//
// IR 结构（简化版，tensor<64xf32> 统一维度）：
//   id=0: %v1(producer) %v2(producer)
//   id=1: %v3(consumer of %v1)
//   id=2: %v5(consumer of %v2), %v6(uses v5 result)
//
// 预期变换：
//   - Group 1: %v1 前插入 producer 逻辑 → buffer write；%v3 前插入 consumer 逻辑 → buffer read
//   - Group 2: %v2 前插入 producer 逻辑 → buffer write；%v5 前插入 consumer 逻辑 → buffer read
//
// RUN: triton-adapter-opt %s --inner-multibuffer 2>&1 | FileCheck %s

module {
  tt.func @inner_multibuffer_test(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant {ssbuffer.nesting_depth = 0 : i32} 0 : index
    %c64 = arith.constant {ssbuffer.nesting_depth = 0 : i32} 64 : index
    %c1024 = arith.constant {ssbuffer.nesting_depth = 0 : i32} 1024 : index
    %cst = arith.constant {ssbuffer.nesting_depth = 0 : i32} dense<0.000000e+00> : tensor<64xf32>

    scope.scope : () -> () {
      // 核内主循环：带 ssbuffer.mainloop，做跨 id 依赖的双缓冲
      %loop = scf.for %iv = %c0 to %c1024 step %c64 iter_args(%acc = %cst) -> tensor<64xf32> {
        // ===== id=0: producers =====
        // %v1: producer，写入 buffer（被 id=1 的 %v3 依赖）
        %v0 = arith.constant {ssbuffer.core_type = "vector", ssbuffer.id = 0 : i32, ssbuffer.nesting_depth = 2 : i32} dense<1.0> : tensor<64xf32>
        %v1 = arith.addf %v0, %v0 {ssbuffer.core_type = "vector", ssbuffer.id = 0 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64xf32>

        // %v2: producer，写入 buffer（被 id=2 的 %v5 依赖）
        %v2 = arith.mulf %acc, %acc {ssbuffer.core_type = "vector", ssbuffer.id = 0 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64xf32>

        // ===== id=1: consumer of %v1 =====
        // %v3: consumer，从 buffer 读取（依赖 id=0 的 %v1）
        %v3 = arith.addf %v1, %v1 {ssbuffer.core_type = "vector", ssbuffer.id = 1 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64xf32>

        // ===== id=2: consumer of %v2 =====
        // %v5: consumer，从 buffer 读取（依赖 id=0 的 %v2）
        // %v6: 独立计算，使用 %v5 的结果（确保 v5 是 depUserMap 的端点）
        %v5 = arith.addf %v2, %v2 {ssbuffer.core_type = "vector", ssbuffer.id = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64xf32>
        %v6 = arith.addf %v5, %v5 {ssbuffer.core_type = "vector", ssbuffer.id = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64xf32>

        scf.yield %v6 : tensor<64xf32>
      } {ssbuffer.mainloop, ssbuffer.nesting_depth = 2 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, ssbuffer.nesting_depth = 1 : i32}

    tt.return
  }
}
