// RUN: triton-opt --analyze-data-flow --verify-diagnostics %s

// CHECK-LABEL: func.func @test_tensor_args_use_after_update
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_tensor_args_use_after_update(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: f32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 11 : i32} 0x7F800000 : f32
    %c1_i32 = arith.constant {ssbuffer.block_id = 11 : i32} 1 : i32
    %cst_0 = arith.constant {ssbuffer.block_id = 11 : i32} 0.000000e+00 : f32
    %c-1_i32 = arith.constant {Undefined, ssbuffer.block_id = 11 : i32} -1 : i32
    %c7_i32 = arith.constant {Undefined, ssbuffer.block_id = 11 : i32} 7 : i32
    %cst_1 = arith.constant {ssbuffer.block_id = 11 : i32} 0.000000e+00 : bf16
    %c64_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 64 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 0 : i32
    %c16384_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 16384 : i32
    %c2048_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 2048 : i32
    %c64 = arith.constant {ssbuffer.block_id = 12 : i32} 64 : index
    %c0 = arith.constant {ssbuffer.block_id = 12 : i32} 0 : index
    %c32_i32 = arith.constant {ssbuffer.block_id = 13 : i32} 32 : i32
    %c6_i32 = arith.constant {ssbuffer.block_id = 13 : i32} 6 : i32
    %c32 = arith.constant {ssbuffer.block_id = 13 : i32} 32 : index
    %cst_2 = arith.constant {ssbuffer.block_id = 9 : i32} dense<[4, 2, 16, 16]> : tensor<4xi64>
    %cst_3 = arith.constant {ssbuffer.block_id = 9 : i32} dense<[32, 4, 16]> : tensor<3xi64>
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<32xf32>
      %1 = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
      %2 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<32x64xf32>
      %3 = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst_0 : f32) outs(%2 : tensor<32x64xf32>) -> tensor<32x64xf32>
      %4 = arith.muli %arg12, %c64_i32 {ssbuffer.block_id = 12 : i32} : i32
      %5 = arith.muli %arg13, %c16384_i32 {ssbuffer.block_id = 12 : i32} : i32
      %6 = arith.muli %arg11, %c32_i32 {ssbuffer.block_id = 13 : i32} : i32
      %7 = arith.muli %arg7, %c32_i32 {ssbuffer.block_id = 13 : i32} : i32
      %8 = arith.muli %arg13, %arg7 {ssbuffer.block_id = 14 : i32} : i32
      %9 = arith.muli %8, %c32_i32 {ssbuffer.block_id = 14 : i32} : i32
      %10 = arith.index_cast %9 {ssbuffer.block_id = 14 : i32} : i32 to index
      %11 = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%arg6 : f32) outs(%2 : tensor<32x64xf32>) -> tensor<32x64xf32>
      %alloc = memref.alloc() {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 1 : i32} : memref<32x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 1 : i32} : memref<32x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
      %12:2 = scf.for %arg14 = %c-1_i32 to %c7_i32 step %c1_i32 iter_args(%arg15 = %1, %arg16 = %3) -> (tensor<32xf32>, tensor<32x64xf32>)  : i32 {
        %13 = arith.subi %c6_i32, %arg14 {ssbuffer.block_id = 8 : i32} : i32
        %14 = arith.muli %13, %c64_i32 {ssbuffer.block_id = 8 : i32} : i32
        %15 = arith.subi %14, %c1_i32 {ssbuffer.block_id = 8 : i32} : i32
        %16 = arith.maxsi %15, %c0_i32 {ssbuffer.block_id = 8 : i32} : i32
        %17 = arith.index_cast %6 {ssbuffer.block_id = 8 : i32} : i32 to index
        %18 = arith.maxsi %17, %c0 {ssbuffer.block_id = 8 : i32} : index
        %19 = arith.subi %c32, %18 {ssbuffer.block_id = 8 : i32} : index
        %20 = arith.maxsi %19, %c0 {ssbuffer.block_id = 8 : i32} : index
        %21 = arith.minsi %20, %c32 {ssbuffer.block_id = 8 : i32} : index
        %22 = arith.index_cast %arg7 {ssbuffer.block_id = 8 : i32} : i32 to index
        %23 = arith.index_cast %14 {ssbuffer.block_id = 8 : i32} : i32 to index
        %24 = arith.maxsi %23, %c0 {ssbuffer.block_id = 8 : i32} : index
        %25 = arith.subi %22, %24 {ssbuffer.block_id = 8 : i32} : index
        %26 = arith.maxsi %25, %c0 {ssbuffer.block_id = 8 : i32} : index
        %27 = arith.minsi %26, %c64 {ssbuffer.block_id = 8 : i32} : index
        %28 = arith.subi %c0_i32, %14 {ssbuffer.block_id = 8 : i32} : i32
        %29 = arith.maxsi %28, %c0_i32 {ssbuffer.block_id = 8 : i32} : i32
        %30 = arith.index_cast %29 {ssbuffer.block_id = 8 : i32} : i32 to index
        %31 = arith.minsi %30, %27 {ssbuffer.block_id = 8 : i32} : index
        %32 = arith.subi %27, %31 {ssbuffer.block_id = 8 : i32} : index
        %33 = arith.subi %c0_i32, %6 {ssbuffer.block_id = 8 : i32} : i32
        %34 = arith.maxsi %33, %c0_i32 {ssbuffer.block_id = 8 : i32} : i32
        %35 = arith.index_cast %34 {ssbuffer.block_id = 8 : i32} : i32 to index
        %36 = arith.minsi %35, %21 {ssbuffer.block_id = 8 : i32} : index
        %37 = arith.subi %21, %36 {ssbuffer.block_id = 8 : i32} : index
        %38 = arith.cmpi slt, %32, %c64 {ssbuffer.block_id = 8 : i32} : index
        %39 = arith.cmpi slt, %37, %c32 {ssbuffer.block_id = 8 : i32} : index
        %40 = arith.ori %38, %39 {ssbuffer.block_id = 8 : i32} : i1
        %41 = arith.muli %16, %c32_i32 {ssbuffer.block_id = 8 : i32} : i32
        %42 = arith.addi %41, %6 {ssbuffer.block_id = 8 : i32} : i32
        %43 = arith.index_cast %7 {ssbuffer.block_id = 8 : i32} : i32 to index
        %44 = arith.index_cast %42 {ssbuffer.block_id = 8 : i32} : i32 to index
        %45 = arith.maxsi %44, %c0 {ssbuffer.block_id = 8 : i32} : index
        %46 = arith.subi %43, %45 {ssbuffer.block_id = 8 : i32} : index
        %47 = arith.maxsi %46, %c0 {ssbuffer.block_id = 8 : i32} : index
        %48 = arith.minsi %47, %c32 {ssbuffer.block_id = 8 : i32} : index
        %49 = arith.subi %c0_i32, %42 {ssbuffer.block_id = 8 : i32} : i32
        %50 = arith.maxsi %49, %c0_i32 {ssbuffer.block_id = 8 : i32} : i32
        %51 = arith.index_cast %50 {ssbuffer.block_id = 8 : i32} : i32 to index
        %52 = arith.minsi %51, %48 {ssbuffer.block_id = 8 : i32} : index
        %53 = arith.subi %48, %52 {ssbuffer.block_id = 8 : i32} : index
        %54 = arith.cmpi slt, %53, %c32 {ssbuffer.block_id = 8 : i32} : index
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 9 : i32} : memref<64x32xbf16>
        %alloc_6 = memref.alloc() {ssbuffer.block_id = 9 : i32} : memref<32xf32>
        %alloc_7 = memref.alloc() {ssbuffer.block_id = 9 : i32} : memref<64x32xf32>
        scf.if %54 {
          linalg.fill {ssbuffer.block_id = 9 : i32} ins(%cst_0 : f32) outs(%alloc_6 : memref<32xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 9 : i32}
        scf.if %40 {
          linalg.fill {ssbuffer.block_id = 9 : i32} ins(%cst_0 : f32) outs(%alloc_7 : memref<64x32xf32>)
          linalg.fill {ssbuffer.block_id = 9 : i32} ins(%cst_1 : bf16) outs(%alloc_5 : memref<64x32xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 9 : i32}
        %55 = arith.maxsi %14, %c0_i32 {ssbuffer.block_id = 9 : i32} : i32
        %56 = arith.index_cast %55 {ssbuffer.block_id = 9 : i32} : i32 to index
        %57 = arith.maxsi %6, %c0_i32 {ssbuffer.block_id = 9 : i32} : i32
        %58 = arith.index_cast %57 {ssbuffer.block_id = 9 : i32} : i32 to index
        %59 = arith.muli %56, %c32 {ssbuffer.block_id = 9 : i32} : index
        %60 = arith.addi %59, %10 {ssbuffer.block_id = 9 : i32} : index
        %61 = arith.addi %60, %58 {ssbuffer.block_id = 9 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%61], sizes: [64, 32], strides: [32, 1] {ssbuffer.block_id = 9 : i32} : memref<?xbf16> to memref<64x32xbf16, strided<[32, 1], offset: ?>>
        %subview = memref.subview %reinterpret_cast[0, 0] [%32, %37] [1, 1] {ssbuffer.block_id = 9 : i32} : memref<64x32xbf16, strided<[32, 1], offset: ?>> to memref<?x?xbf16, strided<[32, 1], offset: ?>>
        %subview_8 = memref.subview %alloc_5[%31, %36] [%32, %37] [1, 1] {ssbuffer.block_id = 9 : i32} : memref<64x32xbf16> to memref<?x?xbf16, strided<[32, 1], offset: ?>>
        memref.copy %subview, %subview_8 {ssbuffer.block_id = 9 : i32} : memref<?x?xbf16, strided<[32, 1], offset: ?>> to memref<?x?xbf16, strided<[32, 1], offset: ?>>
        %62 = bufferization.to_tensor %alloc_5 restrict writable {ssbuffer.block_id = 9 : i32} : memref<64x32xbf16>
        %63 = tensor.empty() {ssbuffer.block_id = 9 : i32} : tensor<32x64xbf16>
        %transposed = linalg.transpose ins(%62 : tensor<64x32xbf16>) outs(%63 : tensor<32x64xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 9 : i32}
        %64 = arith.extf %transposed {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32x64xbf16> to tensor<32x64xf32>
        %65 = arith.mulf %64, %11 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32x64xf32>
        %66 = arith.truncf %65 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32x64xf32> to tensor<32x64xbf16>
        %67 = arith.maxsi %42, %c0_i32 {ssbuffer.block_id = 9 : i32} : i32
        %68 = arith.index_cast %67 {ssbuffer.block_id = 9 : i32} : i32 to index
        %69 = arith.addi %68, %10 {ssbuffer.block_id = 9 : i32} : index
        %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [32], strides: [1] {ssbuffer.block_id = 9 : i32} : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
        %subview_10 = memref.subview %reinterpret_cast_9[0] [%53] [1] {ssbuffer.block_id = 9 : i32} : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %subview_11 = memref.subview %alloc_6[%52] [%53] [1] {ssbuffer.block_id = 9 : i32} : memref<32xf32> to memref<?xf32, strided<[1], offset: ?>>
        memref.copy %subview_10, %subview_11 {ssbuffer.block_id = 9 : i32} : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %70 = bufferization.to_tensor %alloc_6 restrict writable {ssbuffer.block_id = 9 : i32} : memref<32xf32>
        %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%61], sizes: [64, 32], strides: [32, 1] {ssbuffer.block_id = 9 : i32} : memref<?xf32> to memref<64x32xf32, strided<[32, 1], offset: ?>>
        %subview_13 = memref.subview %reinterpret_cast_12[0, 0] [%32, %37] [1, 1] {ssbuffer.block_id = 9 : i32} : memref<64x32xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
        %subview_14 = memref.subview %alloc_7[%31, %36] [%32, %37] [1, 1] {ssbuffer.block_id = 9 : i32} : memref<64x32xf32> to memref<?x?xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_13, %subview_14 {ssbuffer.block_id = 9 : i32} : memref<?x?xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
        %71 = bufferization.to_tensor %alloc_7 restrict writable {ssbuffer.block_id = 9 : i32} : memref<64x32xf32>
        %transposed_15 = linalg.transpose ins(%71 : tensor<64x32xf32>) outs(%2 : tensor<32x64xf32>) permutation = [1, 0]  {ssbuffer.block_id = 9 : i32}
        %broadcasted = linalg.broadcast ins(%70 : tensor<32xf32>) outs(%2 : tensor<32x64xf32>) dimensions = [1]  {ssbuffer.block_id = 9 : i32}
        %72 = arith.subf %broadcasted, %transposed_15 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32x64xf32>
        %73 = math.exp %72 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32x64xf32>
        %74 = arith.extf %66 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32x64xbf16> to tensor<32x64xf32>
        %75 = arith.mulf %74, %73 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32x64xf32>
        %76 = arith.truncf %75 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32x64xf32> to tensor<32x64xbf16>
        %reshape = tensor.reshape %76(%cst_3) {ssbuffer.block_id = 9 : i32} : (tensor<32x64xbf16>, tensor<3xi64>) -> tensor<32x4x16xbf16>
        %77 = tensor.empty() {ssbuffer.block_id = 9 : i32} : tensor<4x32x16xbf16>
        %transposed_16 = linalg.transpose ins(%reshape : tensor<32x4x16xbf16>) outs(%77 : tensor<4x32x16xbf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 9 : i32}
        %reshape_17 = tensor.reshape %transposed_16(%cst_2) {ssbuffer.block_id = 9 : i32} : (tensor<4x32x16xbf16>, tensor<4xi64>) -> tensor<4x2x16x16xbf16>
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.copy ins(%reshape_17 : tensor<4x2x16x16xbf16>) outs(%alloc : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
        %memspacecast = memref.memory_space_cast %alloc_4 {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<32x64xf32, #hivm.address_space<ub>> to memref<32x64xf32>
        %78 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<32x64xf32>
        %79 = arith.maxsi %4, %c0_i32 {ssbuffer.block_id = 10 : i32} : i32
        %80 = arith.index_cast %79 {ssbuffer.block_id = 10 : i32} : i32 to index
        %81 = arith.muli %13, %c2048_i32 {ssbuffer.block_id = 10 : i32} : i32
        %82 = arith.index_cast %5 {ssbuffer.block_id = 10 : i32} : i32 to index
        %83 = arith.index_cast %81 {ssbuffer.block_id = 10 : i32} : i32 to index
        %84 = arith.addi %82, %83 {ssbuffer.block_id = 10 : i32} : index
        %85 = arith.muli %58, %c64 {ssbuffer.block_id = 10 : i32} : index
        %86 = arith.addi %85, %84 {ssbuffer.block_id = 10 : i32} : index
        %87 = arith.addi %86, %80 {ssbuffer.block_id = 10 : i32} : index
        %reinterpret_cast_18 = memref.reinterpret_cast %arg5 to offset: [%87], sizes: [32, 64], strides: [64, 1] {ssbuffer.block_id = 10 : i32} : memref<?xbf16> to memref<32x64xbf16, strided<[64, 1], offset: ?>>
        %88 = arith.index_cast %4 {ssbuffer.block_id = 10 : i32} : i32 to index
        %89 = arith.maxsi %88, %c0 {ssbuffer.block_id = 10 : i32} : index
        %90 = arith.subi %c64, %89 {ssbuffer.block_id = 10 : i32} : index
        %91 = arith.maxsi %90, %c0 {ssbuffer.block_id = 10 : i32} : index
        %92 = arith.minsi %91, %c64 {ssbuffer.block_id = 10 : i32} : index
        %93 = arith.subi %c0_i32, %4 {ssbuffer.block_id = 10 : i32} : i32
        %94 = arith.maxsi %93, %c0_i32 {ssbuffer.block_id = 10 : i32} : i32
        %95 = arith.index_cast %94 {ssbuffer.block_id = 10 : i32} : i32 to index
        %96 = arith.minsi %95, %92 {ssbuffer.block_id = 10 : i32} : index
        %97 = arith.subi %92, %96 {ssbuffer.block_id = 10 : i32} : index
        %98 = arith.truncf %arg16 {DataUse, ssbuffer.block_id = 10 : i32} : tensor<32x64xf32> to tensor<32x64xbf16>
        %extracted_slice = tensor.extract_slice %98[%36, %96] [%37, %97] [1, 1] {ssbuffer.block_id = 10 : i32} : tensor<32x64xbf16> to tensor<?x?xbf16>
        %subview_19 = memref.subview %reinterpret_cast_18[0, 0] [%37, %97] [1, 1] {ssbuffer.block_id = 10 : i32} : memref<32x64xbf16, strided<[64, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_19 {ssbuffer.block_id = 10 : i32} : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[64, 1], offset: ?>>) -> ()
        %99 = arith.subf %70, %arg15 {DataUse, ssbuffer.block_id = 10 : i32} : tensor<32xf32>
        %100 = math.exp %99 {DataUse, ssbuffer.block_id = 10 : i32} : tensor<32xf32>
        %broadcasted_20 = linalg.broadcast ins(%100 : tensor<32xf32>) outs(%2 : tensor<32x64xf32>) dimensions = [1]  {ssbuffer.block_id = 10 : i32}
        %101 = arith.mulf %arg16, %broadcasted_20 {DataUse, ssbuffer.block_id = 10 : i32} : tensor<32x64xf32>
        %102 = arith.addf %78, %101 {ssbuffer.add_from_matmul, ssbuffer.block_id = 10 : i32} : tensor<32x64xf32>
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
        scf.yield %70, %102 : tensor<32xf32>, tensor<32x64xf32>
      } {Undefined, ssbuffer.block_id = 15 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %0 = arith.muli %arg13, %arg7 {ssbuffer.block_id = 4 : i32} : i32
      %1 = arith.muli %0, %c64_i32 {ssbuffer.block_id = 4 : i32} : i32
      %2 = arith.index_cast %1 {ssbuffer.block_id = 4 : i32} : i32 to index
      %3 = arith.muli %arg12, %c64_i32 {ssbuffer.block_id = 4 : i32} : i32
      %alloc = memref.alloc() {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 1 : i32} : memref<32x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 1 : i32} : memref<32x64xf32, #hivm.address_space<ub>>
      scf.for %arg14 = %c-1_i32 to %c7_i32 step %c1_i32  : i32 {
        %4 = arith.subi %c6_i32, %arg14 {ssbuffer.block_id = 2 : i32} : i32
        %5 = arith.muli %4, %c64_i32 {ssbuffer.block_id = 2 : i32} : i32
        %6 = arith.index_cast %arg7 {ssbuffer.block_id = 2 : i32} : i32 to index
        %7 = arith.index_cast %5 {ssbuffer.block_id = 2 : i32} : i32 to index
        %8 = arith.maxsi %7, %c0 {ssbuffer.block_id = 2 : i32} : index
        %9 = arith.subi %6, %8 {ssbuffer.block_id = 2 : i32} : index
        %10 = arith.maxsi %9, %c0 {ssbuffer.block_id = 2 : i32} : index
        %11 = arith.minsi %10, %c64 {ssbuffer.block_id = 2 : i32} : index
        %12 = arith.subi %c0_i32, %5 {ssbuffer.block_id = 2 : i32} : i32
        %13 = arith.maxsi %12, %c0_i32 {ssbuffer.block_id = 2 : i32} : i32
        %14 = arith.index_cast %13 {ssbuffer.block_id = 2 : i32} : i32 to index
        %15 = arith.minsi %14, %11 {ssbuffer.block_id = 2 : i32} : index
        %16 = arith.subi %11, %15 {ssbuffer.block_id = 2 : i32} : index
        %17 = arith.cmpi slt, %16, %c64 {ssbuffer.block_id = 2 : i32} : index
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 2 : i32} : memref<64x64xbf16>
        %18 = arith.index_cast %3 {ssbuffer.block_id = 2 : i32} : i32 to index
        %19 = arith.maxsi %18, %c0 {ssbuffer.block_id = 2 : i32} : index
        %20 = arith.subi %c64, %19 {ssbuffer.block_id = 2 : i32} : index
        %21 = arith.maxsi %20, %c0 {ssbuffer.block_id = 2 : i32} : index
        %22 = arith.minsi %21, %c64 {ssbuffer.block_id = 2 : i32} : index
        %23 = arith.subi %c0_i32, %3 {ssbuffer.block_id = 2 : i32} : i32
        %24 = arith.maxsi %23, %c0_i32 {ssbuffer.block_id = 2 : i32} : i32
        %25 = arith.index_cast %24 {ssbuffer.block_id = 2 : i32} : i32 to index
        %26 = arith.minsi %25, %22 {ssbuffer.block_id = 2 : i32} : index
        %27 = arith.subi %22, %26 {ssbuffer.block_id = 2 : i32} : index
        %28 = arith.cmpi slt, %27, %c64 {ssbuffer.block_id = 2 : i32} : index
        %29 = arith.ori %17, %28 {ssbuffer.block_id = 2 : i32} : i1
        scf.if %29 {
          linalg.fill {ssbuffer.block_id = 2 : i32} ins(%cst_1 : bf16) outs(%alloc_5 : memref<64x64xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 2 : i32}
        %30 = arith.maxsi %5, %c0_i32 {ssbuffer.block_id = 2 : i32} : i32
        %31 = arith.index_cast %30 {ssbuffer.block_id = 2 : i32} : i32 to index
        %32 = arith.maxsi %3, %c0_i32 {ssbuffer.block_id = 2 : i32} : i32
        %33 = arith.index_cast %32 {ssbuffer.block_id = 2 : i32} : i32 to index
        %34 = arith.muli %31, %c64 {ssbuffer.block_id = 2 : i32} : index
        %35 = arith.addi %34, %2 {ssbuffer.block_id = 2 : i32} : index
        %36 = arith.addi %35, %33 {ssbuffer.block_id = 2 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%36], sizes: [64, 64], strides: [64, 1] {ssbuffer.block_id = 2 : i32} : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
        %subview = memref.subview %reinterpret_cast[0, 0] [%16, %27] [1, 1] {ssbuffer.block_id = 2 : i32} : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        %subview_6 = memref.subview %alloc_5[%15, %26] [%16, %27] [1, 1] {ssbuffer.block_id = 2 : i32} : memref<64x64xbf16> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        memref.copy %subview, %subview_6 {ssbuffer.block_id = 2 : i32} : memref<?x?xbf16, strided<[64, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        %37 = bufferization.to_tensor %alloc_5 restrict writable {ssbuffer.block_id = 2 : i32} : memref<64x64xbf16>
        %38 = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<32x64xf32>
        %39 = linalg.fill {ssbuffer.block_id = 2 : i32} ins(%cst_0 : f32) outs(%38 : tensor<32x64xf32>) -> tensor<32x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        %40 = hivm.hir.convert_layout %alloc output_shape [32, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<32x64xbf16, #hivm.address_space<cbuf>>
        %memspacecast = memref.memory_space_cast %40 {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32} : memref<32x64xbf16, #hivm.address_space<cbuf>> to memref<32x64xbf16>
        %41 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32} : memref<32x64xbf16>
        %42 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32} ins(%41, %37 : tensor<32x64xbf16>, tensor<64x64xbf16>) outs(%39 : tensor<32x64xf32>) -> tensor<32x64xf32>
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32} ins(%42 : tensor<32x64xf32>) outs(%alloc_4 : memref<32x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
      } {Undefined, ssbuffer.block_id = 15 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}