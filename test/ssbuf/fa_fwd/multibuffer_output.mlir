// before addifcontrols:
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant dense<[16, 8, 16, 8]> : tensor<4xi64>
    %cst_0 = arith.constant dense<[128, 16, 8]> : tensor<3xi64>
    %cst_1 = arith.constant dense<[8, 8, 16, 16]> : tensor<4xi64>
    %cst_2 = arith.constant dense<[128, 8, 16]> : tensor<3xi64>
    %c28_i32 = arith.constant 28 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c128_i32 = arith.constant 128 : i32
    %c131072_i64 = arith.constant 131072 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %cst_4 = arith.constant 5.000000e-01 : f32
    %cst_5 = arith.constant 0xFF800000 : f32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    %alloc_7 = memref.alloc() : memref<16x8x16x8xf32, #hivm.address_space<cbuf>>
    %alloc_8 = memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %alloc_9 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 10 : i32} : tensor<128x128xf32>
      %1 = linalg.fill {ssbuffer.block_id = 10 : i32} ins(%cst_3 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %2 = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst_4 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %3 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<128xf32>
      %4 = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst_5 : f32) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
      %5 = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst_6 : f32) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
      scf.for %arg15 = %arg12 to %c8192_i32 step %c28_i32  : i32 {
        %6 = arith.divsi %arg15, %c8_i32 {ssbuffer.block_id = 8 : i32} : i32
        %7 = arith.remsi %arg15, %c8_i32 {ssbuffer.block_id = 8 : i32} : i32
        %8 = arith.divsi %6, %c8_i32 {ssbuffer.block_id = 8 : i32} : i32
        %9 = arith.remsi %6, %c8_i32 {ssbuffer.block_id = 8 : i32} : i32
        %10 = arith.extsi %8 {ssbuffer.block_id = 8 : i32} : i32 to i64
        %11 = arith.muli %10, %c1048576_i64 {ssbuffer.block_id = 8 : i32} : i64
        %12 = arith.extsi %9 {ssbuffer.block_id = 8 : i32} : i32 to i64
        %13 = arith.muli %12, %c131072_i64 {ssbuffer.block_id = 8 : i32} : i64
        %14 = arith.addi %11, %13 {ssbuffer.block_id = 8 : i32} : i64
        %15 = arith.index_cast %14 {ssbuffer.block_id = 8 : i32} : i64 to index
        %16 = arith.muli %7, %c128_i32 {ssbuffer.block_id = 8 : i32} : i32
        %17 = arith.maxsi %16, %c0_i32 {ssbuffer.block_id = 8 : i32} : i32
        %18 = arith.index_cast %17 {ssbuffer.block_id = 8 : i32} : i32 to index
        %19 = arith.muli %18, %c128 {ssbuffer.block_id = 8 : i32} : index
        %20 = arith.addi %19, %15 {ssbuffer.block_id = 8 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%20], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 8 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
        %21 = arith.muli %6, %c1024_i32 {ssbuffer.block_id = 8 : i32} : i32
        %22 = arith.index_cast %21 {ssbuffer.block_id = 8 : i32} : i32 to index
        %23 = arith.index_cast %16 {ssbuffer.block_id = 8 : i32} : i32 to index
        %24 = arith.addi %22, %23 {ssbuffer.block_id = 8 : i32} : index
        %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [128], strides: [1] {ssbuffer.block_id = 8 : i32} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 40 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
        hivm.hir.sync_block_set {ssbuffer.block_id = 40 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %alloc_11 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
        %memspacecast = memref.memory_space_cast %alloc_11 : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %25 = bufferization.to_tensor %memspacecast : memref<128x128xf32>
        %alloc_12 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
        %memspacecast_13 = memref.memory_space_cast %alloc_12 : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %26 = bufferization.to_tensor %memspacecast_13 : memref<128x128xf32>
        %alloc_14 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
        %memspacecast_15 = memref.memory_space_cast %alloc_14 : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %27 = bufferization.to_tensor %memspacecast_15 : memref<128x128xf32>
        %alloc_16 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
        %memspacecast_17 = memref.memory_space_cast %alloc_16 : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %28 = bufferization.to_tensor %memspacecast_17 : memref<128x128xf32>
        %alloc_18 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
        %memspacecast_19 = memref.memory_space_cast %alloc_18 : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
        %29 = bufferization.to_tensor %memspacecast_19 : memref<128xf32>
        %alloc_20 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
        %memspacecast_21 = memref.memory_space_cast %alloc_20 : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
        %30 = bufferization.to_tensor %memspacecast_21 : memref<128xf32>
        %alloc_22 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
        %memspacecast_23 = memref.memory_space_cast %alloc_22 : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
        %31 = bufferization.to_tensor %memspacecast_23 : memref<128xf32>
        %alloc_24 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
        %memspacecast_25 = memref.memory_space_cast %alloc_24 : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
        %32 = bufferization.to_tensor %memspacecast_25 : memref<128xf32>
        %33:3 = scf.for %arg16 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg17 = %5, %arg18 = %1, %arg19 = %4) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>)  : i32 {
          %38 = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_3 : f32) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
          %39 = arith.subi %arg16, %c0_i32 {ssbuffer.block_id = 5 : i32} : i32
          %40 = arith.divui %39, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          %c2_i32 = arith.constant {ssbuffer.block_id = 5 : i32} 2 : i32
          %41 = arith.remsi %40, %c2_i32 {ssbuffer.block_id = 5 : i32} : i32
          %c0_i32_26 = arith.constant {ssbuffer.block_id = 5 : i32} 0 : i32
          %42 = arith.cmpi eq, %41, %c0_i32_26 {ssbuffer.block_id = 5 : i32} : i32
          scf.if %42 {
            bufferization.materialize_in_destination %38 in writable %memspacecast_19 {ssbuffer.block_id = 5 : i32} : (tensor<128xf32>, memref<128xf32>) -> ()
          } {ssbuffer.block_id = 5 : i32}
          %c1_i32 = arith.constant {ssbuffer.block_id = 5 : i32} 1 : i32
          %43 = arith.cmpi eq, %41, %c1_i32 {ssbuffer.block_id = 5 : i32} : i32
          scf.if %43 {
            bufferization.materialize_in_destination %38 in writable %memspacecast_21 {ssbuffer.block_id = 5 : i32} : (tensor<128xf32>, memref<128xf32>) -> ()
          } {ssbuffer.block_id = 5 : i32}
          hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %memspacecast_27 = memref.memory_space_cast %alloc {ssbuffer.block_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %44 = bufferization.to_tensor %memspacecast_27 restrict writable {ssbuffer.block_id = 5 : i32} : memref<128x128xf32>
          %45 = arith.mulf %44, %2 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
          %reduced = linalg.reduce ins(%45 : tensor<128x128xf32>) outs(%4 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 5 : i32}
            (%in: f32, %init: f32) {
              %98 = arith.maximumf %in, %init : f32
              linalg.yield %98 : f32
            }
          %46 = arith.maximumf %arg19, %reduced {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
          %47 = arith.subi %arg16, %c0_i32 {ssbuffer.block_id = 5 : i32} : i32
          %48 = arith.divui %47, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          %c2_i32_28 = arith.constant {ssbuffer.block_id = 5 : i32} 2 : i32
          %49 = arith.remsi %48, %c2_i32_28 {ssbuffer.block_id = 5 : i32} : i32
          %c0_i32_29 = arith.constant {ssbuffer.block_id = 5 : i32} 0 : i32
          %50 = arith.cmpi eq, %49, %c0_i32_29 {ssbuffer.block_id = 5 : i32} : i32
          scf.if %50 {
            bufferization.materialize_in_destination %46 in writable %memspacecast_23 {ssbuffer.block_id = 5 : i32} : (tensor<128xf32>, memref<128xf32>) -> ()
          } {ssbuffer.block_id = 5 : i32}
          %c1_i32_30 = arith.constant {ssbuffer.block_id = 5 : i32} 1 : i32
          %51 = arith.cmpi eq, %49, %c1_i32_30 {ssbuffer.block_id = 5 : i32} : i32
          scf.if %51 {
            bufferization.materialize_in_destination %46 in writable %memspacecast_25 {ssbuffer.block_id = 5 : i32} : (tensor<128xf32>, memref<128xf32>) -> ()
          } {ssbuffer.block_id = 5 : i32}
          %broadcasted_31 = linalg.broadcast ins(%46 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 5 : i32}
          %52 = arith.subf %45, %broadcasted_31 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
          %53 = math.exp %52 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
          %54 = arith.subi %arg16, %c0_i32 {ssbuffer.block_id = 5 : i32} : i32
          %55 = arith.divui %54, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          %c2_i32_32 = arith.constant {ssbuffer.block_id = 5 : i32} 2 : i32
          %56 = arith.remsi %55, %c2_i32_32 {ssbuffer.block_id = 5 : i32} : i32
          %c0_i32_33 = arith.constant {ssbuffer.block_id = 5 : i32} 0 : i32
          %57 = arith.cmpi eq, %56, %c0_i32_33 {ssbuffer.block_id = 5 : i32} : i32
          scf.if %57 {
            bufferization.materialize_in_destination %53 in writable %memspacecast_15 {ssbuffer.block_id = 5 : i32} : (tensor<128x128xf32>, memref<128x128xf32>) -> ()
          } {ssbuffer.block_id = 5 : i32}
          %c1_i32_34 = arith.constant {ssbuffer.block_id = 5 : i32} 1 : i32
          %58 = arith.cmpi eq, %56, %c1_i32_34 {ssbuffer.block_id = 5 : i32} : i32
          scf.if %58 {
            bufferization.materialize_in_destination %53 in writable %memspacecast_17 {ssbuffer.block_id = 5 : i32} : (tensor<128x128xf32>, memref<128x128xf32>) -> ()
          } {ssbuffer.block_id = 5 : i32}
          %59 = arith.truncf %53 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
          %reshape = tensor.reshape %59(%cst_2) {ssbuffer.block_id = 5 : i32} : (tensor<128x128xf16>, tensor<3xi64>) -> tensor<128x8x16xf16>
          annotation.mark %reshape {ssbuffer.block_id = 5 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<128x8x16xf16>
          %60 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x128x16xf16>
          %transposed = linalg.transpose ins(%reshape : tensor<128x8x16xf16>) outs(%60 : tensor<8x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 5 : i32}
          %reshape_35 = tensor.reshape %transposed(%cst_1) {ssbuffer.block_id = 5 : i32} : (tensor<8x128x16xf16>, tensor<4xi64>) -> tensor<8x8x16x16xf16>
          annotation.mark %reshape_35 {ssbuffer.block_id = 5 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<8x8x16x16xf16>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          hivm.hir.copy ins(%reshape_35 : tensor<8x8x16x16xf16>) outs(%alloc_8 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32}
          hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
          hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
          %61 = arith.subi %arg16, %c0_i32 {ssbuffer.block_id = 6 : i32} : i32
          %62 = arith.divui %61, %c128_i32 {ssbuffer.block_id = 6 : i32} : i32
          %c2_i32_36 = arith.constant {ssbuffer.block_id = 6 : i32} 2 : i32
          %c1_i32_37 = arith.constant {ssbuffer.block_id = 6 : i32} 1 : i32
          %63 = arith.addi %62, %c1_i32_37 {ssbuffer.block_id = 6 : i32} : i32
          %64 = arith.remsi %63, %c2_i32_36 {ssbuffer.block_id = 6 : i32} : i32
          %c0_i32_38 = arith.constant {ssbuffer.block_id = 6 : i32} 0 : i32
          %65 = arith.cmpi eq, %64, %c0_i32_38 {ssbuffer.block_id = 6 : i32} : i32
          %66 = scf.if %65 -> (tensor<128x128xf32>) {
            %98 = bufferization.to_tensor %memspacecast_15 {ssbuffer.block_id = 6 : i32} : memref<128x128xf32>
            scf.yield %98 : tensor<128x128xf32>
          } else {
            %98 = bufferization.to_tensor %memspacecast_17 {ssbuffer.block_id = 6 : i32} : memref<128x128xf32>
            scf.yield %98 : tensor<128x128xf32>
          } {ssbuffer.block_id = 6 : i32}
          %67 = arith.subi %arg16, %c0_i32 {ssbuffer.block_id = 6 : i32} : i32
          %68 = arith.divui %67, %c128_i32 {ssbuffer.block_id = 6 : i32} : i32
          %c2_i32_39 = arith.constant {ssbuffer.block_id = 6 : i32} 2 : i32
          %c1_i32_40 = arith.constant {ssbuffer.block_id = 6 : i32} 1 : i32
          %69 = arith.addi %68, %c1_i32_40 {ssbuffer.block_id = 6 : i32} : i32
          %70 = arith.remsi %69, %c2_i32_39 {ssbuffer.block_id = 6 : i32} : i32
          %c0_i32_41 = arith.constant {ssbuffer.block_id = 6 : i32} 0 : i32
          %71 = arith.cmpi eq, %70, %c0_i32_41 {ssbuffer.block_id = 6 : i32} : i32
          %72 = scf.if %71 -> (tensor<128xf32>) {
            %98 = bufferization.to_tensor %memspacecast_19 {ssbuffer.block_id = 6 : i32} : memref<128xf32>
            scf.yield %98 : tensor<128xf32>
          } else {
            %98 = bufferization.to_tensor %memspacecast_21 {ssbuffer.block_id = 6 : i32} : memref<128xf32>
            scf.yield %98 : tensor<128xf32>
          } {ssbuffer.block_id = 6 : i32}
          %reduced_42 = linalg.reduce ins(%66 : tensor<128x128xf32>) outs(%72 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 6 : i32}
            (%in: f32, %init: f32) {
              %98 = arith.addf %in, %init : f32
              linalg.yield %98 : f32
            }
          %73 = arith.subi %arg16, %c0_i32 {ssbuffer.block_id = 6 : i32} : i32
          %74 = arith.divui %73, %c128_i32 {ssbuffer.block_id = 6 : i32} : i32
          %c2_i32_43 = arith.constant {ssbuffer.block_id = 6 : i32} 2 : i32
          %c1_i32_44 = arith.constant {ssbuffer.block_id = 6 : i32} 1 : i32
          %75 = arith.addi %74, %c1_i32_44 {ssbuffer.block_id = 6 : i32} : i32
          %76 = arith.remsi %75, %c2_i32_43 {ssbuffer.block_id = 6 : i32} : i32
          %c0_i32_45 = arith.constant {ssbuffer.block_id = 6 : i32} 0 : i32
          %77 = arith.cmpi eq, %76, %c0_i32_45 {ssbuffer.block_id = 6 : i32} : i32
          %78 = scf.if %77 -> (tensor<128xf32>) {
            %98 = bufferization.to_tensor %memspacecast_23 {ssbuffer.block_id = 6 : i32} : memref<128xf32>
            scf.yield %98 : tensor<128xf32>
          } else {
            %98 = bufferization.to_tensor %memspacecast_25 {ssbuffer.block_id = 6 : i32} : memref<128xf32>
            scf.yield %98 : tensor<128xf32>
          } {ssbuffer.block_id = 6 : i32}
          %79 = arith.subf %arg19, %78 {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
          %80 = math.exp %79 {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
          %81 = arith.mulf %arg17, %80 {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
          %82 = arith.addf %81, %reduced_42 {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
          %broadcasted_46 = linalg.broadcast ins(%80 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 6 : i32}
          %83 = arith.subi %arg16, %c0_i32 {ssbuffer.block_id = 6 : i32} : i32
          %84 = arith.divui %83, %c128_i32 {ssbuffer.block_id = 6 : i32} : i32
          %c2_i32_47 = arith.constant {ssbuffer.block_id = 6 : i32} 2 : i32
          %85 = arith.remsi %84, %c2_i32_47 {ssbuffer.block_id = 6 : i32} : i32
          %c0_i32_48 = arith.constant {ssbuffer.block_id = 6 : i32} 0 : i32
          %86 = arith.cmpi eq, %85, %c0_i32_48 {ssbuffer.block_id = 6 : i32} : i32
          scf.if %86 {
            bufferization.materialize_in_destination %broadcasted_46 in writable %memspacecast {ssbuffer.block_id = 6 : i32} : (tensor<128x128xf32>, memref<128x128xf32>) -> ()
          } {ssbuffer.block_id = 6 : i32}
          %c1_i32_49 = arith.constant {ssbuffer.block_id = 6 : i32} 1 : i32
          %87 = arith.cmpi eq, %85, %c1_i32_49 {ssbuffer.block_id = 6 : i32} : i32
          scf.if %87 {
            bufferization.materialize_in_destination %broadcasted_46 in writable %memspacecast_13 {ssbuffer.block_id = 6 : i32} : (tensor<128x128xf32>, memref<128x128xf32>) -> ()
          } {ssbuffer.block_id = 6 : i32}
          %88 = arith.subi %arg16, %c0_i32 {ssbuffer.block_id = 7 : i32} : i32
          %89 = arith.divui %88, %c128_i32 {ssbuffer.block_id = 7 : i32} : i32
          %c2_i32_50 = arith.constant {ssbuffer.block_id = 7 : i32} 2 : i32
          %c1_i32_51 = arith.constant {ssbuffer.block_id = 7 : i32} 1 : i32
          %90 = arith.addi %89, %c1_i32_51 {ssbuffer.block_id = 7 : i32} : i32
          %91 = arith.remsi %90, %c2_i32_50 {ssbuffer.block_id = 7 : i32} : i32
          %c0_i32_52 = arith.constant {ssbuffer.block_id = 7 : i32} 0 : i32
          %92 = arith.cmpi eq, %91, %c0_i32_52 {ssbuffer.block_id = 7 : i32} : i32
          %93 = scf.if %92 -> (tensor<128x128xf32>) {
            %98 = bufferization.to_tensor %memspacecast {ssbuffer.block_id = 7 : i32} : memref<128x128xf32>
            scf.yield %98 : tensor<128x128xf32>
          } else {
            %98 = bufferization.to_tensor %memspacecast_13 {ssbuffer.block_id = 7 : i32} : memref<128x128xf32>
            scf.yield %98 : tensor<128x128xf32>
          } {ssbuffer.block_id = 7 : i32}
          %94 = arith.mulf %arg18, %93 {ssbuffer.block_id = 7 : i32} : tensor<128x128xf32>
          %reshape_53 = tensor.reshape %94(%cst_0) {ssbuffer.block_id = 7 : i32} : (tensor<128x128xf32>, tensor<3xi64>) -> tensor<128x16x8xf32>
          annotation.mark %reshape_53 {ssbuffer.block_id = 7 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<128x16x8xf32>
          %95 = tensor.empty() {ssbuffer.block_id = 7 : i32} : tensor<16x128x8xf32>
          %transposed_54 = linalg.transpose ins(%reshape_53 : tensor<128x16x8xf32>) outs(%95 : tensor<16x128x8xf32>) permutation = [1, 0, 2]  {ssbuffer.block_id = 7 : i32}
          %reshape_55 = tensor.reshape %transposed_54(%cst) {ssbuffer.block_id = 7 : i32} : (tensor<16x128x8xf32>, tensor<4xi64>) -> tensor<16x8x16x8xf32>
          annotation.mark %reshape_55 {ssbuffer.block_id = 7 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<16x8x16x8xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 7 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
          hivm.hir.copy ins(%reshape_55 : tensor<16x8x16x8xf32>) outs(%alloc_7 : memref<16x8x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 7 : i32}
          hivm.hir.sync_block_set {ssbuffer.block_id = 7 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          hivm.hir.sync_block_wait {ssbuffer.block_id = 70 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
          %memspacecast_56 = memref.memory_space_cast %alloc_9 {ssbuffer.block_id = 70 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %96 = bufferization.to_tensor %memspacecast_56 restrict writable {ssbuffer.block_id = 70 : i32} : memref<128x128xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 70 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
          %97 = math.exp %96 {ssbuffer.block_id = 70 : i32} : tensor<128x128xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 70 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
          scf.yield {ssbuffer.block_id = 60 : i32} %82, %97, %46 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>
        } {ssbuffer.block_id = 40 : i32, ssbuffer.mainloop}
        hivm.hir.sync_block_wait {ssbuffer.block_id = 40 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        hivm.hir.sync_block_wait {ssbuffer.block_id = 40 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        %34 = math.log %33#0 {ssbuffer.block_id = 9 : i32} : tensor<128xf32>
        %35 = arith.addf %33#2, %34 {ssbuffer.block_id = 9 : i32} : tensor<128xf32>
        %broadcasted = linalg.broadcast ins(%33#0 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 9 : i32}
        %36 = arith.divf %33#1, %broadcasted {ssbuffer.block_id = 9 : i32} : tensor<128x128xf32>
        bufferization.materialize_in_destination %35 in writable %reinterpret_cast_10 {ssbuffer.block_id = 9 : i32} : (tensor<128xf32>, memref<128xf32, strided<[1], offset: ?>>) -> ()
        %37 = arith.truncf %36 {ssbuffer.block_id = 9 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
        bufferization.materialize_in_destination %37 in writable %reinterpret_cast {ssbuffer.block_id = 9 : i32} : (tensor<128x128xf16>, memref<128x128xf16, strided<[128, 1], offset: ?>>) -> ()
      } {ssbuffer.block_id = 41 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 50 : i32} : tensor<128x128xf32>
      %1 = linalg.fill {ssbuffer.block_id = 50 : i32} ins(%cst_3 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.for %arg15 = %arg12 to %c8192_i32 step %c28_i32  : i32 {
        %2 = arith.divsi %arg15, %c8_i32 {ssbuffer.block_id = 2 : i32} : i32
        %3 = arith.remsi %arg15, %c8_i32 {ssbuffer.block_id = 2 : i32} : i32
        %4 = arith.divsi %2, %c8_i32 {ssbuffer.block_id = 2 : i32} : i32
        %5 = arith.remsi %2, %c8_i32 {ssbuffer.block_id = 2 : i32} : i32
        %6 = arith.extsi %4 {ssbuffer.block_id = 2 : i32} : i32 to i64
        %7 = arith.muli %6, %c1048576_i64 {ssbuffer.block_id = 2 : i32} : i64
        %8 = arith.extsi %5 {ssbuffer.block_id = 2 : i32} : i32 to i64
        %9 = arith.muli %8, %c131072_i64 {ssbuffer.block_id = 2 : i32} : i64
        %10 = arith.addi %7, %9 {ssbuffer.block_id = 2 : i32} : i64
        %11 = arith.index_cast %10 {ssbuffer.block_id = 2 : i32} : i64 to index
        %12 = arith.muli %3, %c128_i32 {ssbuffer.block_id = 2 : i32} : i32
        %13 = arith.maxsi %12, %c0_i32 {ssbuffer.block_id = 2 : i32} : i32
        %14 = arith.index_cast %13 {ssbuffer.block_id = 2 : i32} : i32 to index
        %15 = arith.muli %14, %c128 {ssbuffer.block_id = 2 : i32} : index
        %16 = arith.addi %15, %11 {ssbuffer.block_id = 2 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 2 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
        %alloc_10 = memref.alloc() {ssbuffer.block_id = 2 : i32} : memref<128x128xf16>
        memref.copy %reinterpret_cast, %alloc_10 {ssbuffer.block_id = 2 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
        %17 = bufferization.to_tensor %alloc_10 restrict writable {ssbuffer.block_id = 2 : i32} : memref<128x128xf16>
        hivm.hir.sync_block_set {ssbuffer.block_id = 40 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.sync_block_set {ssbuffer.block_id = 40 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        %18:2 = scf.for %arg16 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
          %19 = arith.index_cast %arg18 {ssbuffer.block_id = 0 : i32} : i32 to index
          %20 = arith.muli %19, %c128 {ssbuffer.block_id = 0 : i32} : index
          %21 = arith.addi %20, %11 {ssbuffer.block_id = 0 : i32} : index
          %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%21], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 0 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
          %alloc_12 = memref.alloc() {ssbuffer.block_id = 0 : i32} : memref<128x128xf16>
          memref.copy %reinterpret_cast_11, %alloc_12 {ssbuffer.block_id = 0 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
          %22 = bufferization.to_tensor %alloc_12 restrict writable {ssbuffer.block_id = 0 : i32} : memref<128x128xf16>
          %23 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf16>
          %transposed = linalg.transpose ins(%22 : tensor<128x128xf16>) outs(%23 : tensor<128x128xf16>) permutation = [1, 0]  {ssbuffer.block_id = 0 : i32}
          %24 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 0 : i32} ins(%17, %transposed : tensor<128x128xf16>, tensor<128x128xf16>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32} ins(%24 : tensor<128x128xf32>) outs(%alloc : memref<128x128xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.block_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %25 = arith.addi %arg17, %c128_i32 {ssbuffer.block_id = 4 : i32} : i32
          %26 = arith.addi %arg18, %c128_i32 {ssbuffer.block_id = 4 : i32} : i32
          hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 2
          %27 = hivm.hir.convert_layout %alloc_7 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32} : (memref<16x8x16x8xf32, #hivm.address_space<cbuf>>) -> memref<128x128xf32, #hivm.address_space<cbuf>>
          %memspacecast = memref.memory_space_cast %27 {ssbuffer.block_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<cbuf>> to memref<128x128xf32>
          %28 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 1 : i32} : memref<128x128xf32>
          %29 = hivm.hir.convert_layout %alloc_8 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
          %memspacecast_13 = memref.memory_space_cast %29 {ssbuffer.block_id = 1 : i32} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
          %30 = bufferization.to_tensor %memspacecast_13 restrict writable {ssbuffer.block_id = 1 : i32} : memref<128x128xf16>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
          %31 = arith.index_cast %arg17 {ssbuffer.block_id = 1 : i32} : i32 to index
          %32 = arith.muli %31, %c128 {ssbuffer.block_id = 1 : i32} : index
          %33 = arith.addi %32, %11 {ssbuffer.block_id = 1 : i32} : index
          %reinterpret_cast_14 = memref.reinterpret_cast %arg4 to offset: [%33], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 1 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
          %alloc_15 = memref.alloc() {ssbuffer.block_id = 1 : i32} : memref<128x128xf16>
          memref.copy %reinterpret_cast_14, %alloc_15 {ssbuffer.block_id = 1 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
          %34 = bufferization.to_tensor %alloc_15 restrict writable {ssbuffer.block_id = 1 : i32} : memref<128x128xf16>
          %35 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 1 : i32} ins(%30, %34 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%28 : tensor<128x128xf32>) -> tensor<128x128xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
          hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32} ins(%35 : tensor<128x128xf32>) outs(%alloc_9 : memref<128x128xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 4
          scf.yield {ssbuffer.block_id = 60 : i32} %25, %26 : i32, i32
        } {ssbuffer.block_id = 40 : i32, ssbuffer.mainloop}
        hivm.hir.sync_block_wait {ssbuffer.block_id = 40 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
        hivm.hir.sync_block_wait {ssbuffer.block_id = 40 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
      } {ssbuffer.block_id = 41 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}