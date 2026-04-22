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
    %alloc_9 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    %alloc_10 = memref.alloc() : memref<16x8x16x8xf32, #hivm.address_space<cbuf>>
    %alloc_11 = memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %alloc_15 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
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
        %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [128], strides: [1] {ssbuffer.block_id = 8 : i32} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 40 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
        hivm.hir.sync_block_set {ssbuffer.block_id = 40 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %25:3 = scf.for %arg16 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg17 = %5, %arg18 = %1, %arg19 = %4) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>)  : i32 {
          %30 = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_3 : f32) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %memspacecast = memref.memory_space_cast %alloc_9 {ssbuffer.block_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %31 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 5 : i32} : memref<128x128xf32>
          %32 = arith.mulf %31, %2 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
          %reduced = linalg.reduce ins(%32 : tensor<128x128xf32>) outs(%4 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 5 : i32}
            (%in: f32, %init: f32) {
              %46 = arith.maximumf %in, %init : f32
              linalg.yield %46 : f32
            }
          %33 = arith.maximumf %arg19, %reduced {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
          %broadcasted_8 = linalg.broadcast ins(%33 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 5 : i32}
          %34 = arith.subf %32, %broadcasted_8 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
          %35 = math.exp %34 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
          %36 = arith.truncf %35 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
          %reshape = tensor.reshape %36(%cst_2) {ssbuffer.block_id = 5 : i32} : (tensor<128x128xf16>, tensor<3xi64>) -> tensor<128x8x16xf16>
          annotation.mark %reshape {ssbuffer.block_id = 5 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<128x8x16xf16>
          %37 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x128x16xf16>
          %transposed = linalg.transpose ins(%reshape : tensor<128x8x16xf16>) outs(%37 : tensor<8x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 5 : i32}
          %reshape_9 = tensor.reshape %transposed(%cst_1) {ssbuffer.block_id = 5 : i32} : (tensor<8x128x16xf16>, tensor<4xi64>) -> tensor<8x8x16x16xf16>
          annotation.mark %reshape_9 {ssbuffer.block_id = 5 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<8x8x16x16xf16>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          hivm.hir.copy ins(%reshape_9 : tensor<8x8x16x16xf16>) outs(%alloc_11 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32}
          hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
          hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
          %reduced_11 = linalg.reduce ins(%35 : tensor<128x128xf32>) outs(%30 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 6 : i32}
            (%in: f32, %init: f32) {
              %46 = arith.addf %in, %init : f32
              linalg.yield %46 : f32
            }
          %38 = arith.subf %arg19, %33 {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
          %39 = math.exp %38 {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
          %40 = arith.mulf %arg17, %39 {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
          %41 = arith.addf %40, %reduced_11 {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
          %broadcasted_12 = linalg.broadcast ins(%39 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 6 : i32}
          %42 = arith.mulf %arg18, %broadcasted_12 {ssbuffer.block_id = 7 : i32} : tensor<128x128xf32>
          %reshape_13 = tensor.reshape %42(%cst_0) {ssbuffer.block_id = 7 : i32} : (tensor<128x128xf32>, tensor<3xi64>) -> tensor<128x16x8xf32>
          annotation.mark %reshape_13 {ssbuffer.block_id = 7 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<128x16x8xf32>
          %43 = tensor.empty() {ssbuffer.block_id = 7 : i32} : tensor<16x128x8xf32>
          %transposed_14 = linalg.transpose ins(%reshape_13 : tensor<128x16x8xf32>) outs(%43 : tensor<16x128x8xf32>) permutation = [1, 0, 2]  {ssbuffer.block_id = 7 : i32}
          %reshape_15 = tensor.reshape %transposed_14(%cst) {ssbuffer.block_id = 7 : i32} : (tensor<16x128x8xf32>, tensor<4xi64>) -> tensor<16x8x16x8xf32>
          annotation.mark %reshape_15 {ssbuffer.block_id = 7 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<16x8x16x8xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 7 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
          hivm.hir.copy ins(%reshape_15 : tensor<16x8x16x8xf32>) outs(%alloc_10 : memref<16x8x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 7 : i32}
          hivm.hir.sync_block_set {ssbuffer.block_id = 7 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          hivm.hir.sync_block_wait {ssbuffer.block_id = 70 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
          %memspacecast_18 = memref.memory_space_cast %alloc_15 {ssbuffer.block_id = 70 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %44 = bufferization.to_tensor %memspacecast_18 restrict writable {ssbuffer.block_id = 70 : i32} : memref<128x128xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 70 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
          %45 = math.exp %44 {ssbuffer.block_id = 70 : i32} : tensor<128x128xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 70 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
          scf.yield {ssbuffer.block_id = 60 : i32} %41, %45, %33 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>
        } {ssbuffer.block_id = 40 : i32, ssbuffer.mainloop}
        hivm.hir.sync_block_wait {ssbuffer.block_id = 40 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        hivm.hir.sync_block_wait {ssbuffer.block_id = 40 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        %26 = math.log %25#0 {ssbuffer.block_id = 9 : i32} : tensor<128xf32>
        %27 = arith.addf %25#2, %26 {ssbuffer.block_id = 9 : i32} : tensor<128xf32>
        %broadcasted = linalg.broadcast ins(%25#0 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 9 : i32}
        %28 = arith.divf %25#1, %broadcasted {ssbuffer.block_id = 9 : i32} : tensor<128x128xf32>
        bufferization.materialize_in_destination %27 in writable %reinterpret_cast_7 {ssbuffer.block_id = 9 : i32} : (tensor<128xf32>, memref<128xf32, strided<[1], offset: ?>>) -> ()
        %29 = arith.truncf %28 {ssbuffer.block_id = 9 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
        bufferization.materialize_in_destination %29 in writable %reinterpret_cast {ssbuffer.block_id = 9 : i32} : (tensor<128x128xf16>, memref<128x128xf16, strided<[128, 1], offset: ?>>) -> ()
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
        %alloc = memref.alloc() {ssbuffer.block_id = 2 : i32} : memref<128x128xf16>
        memref.copy %reinterpret_cast, %alloc {ssbuffer.block_id = 2 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
        %17 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 2 : i32} : memref<128x128xf16>
        hivm.hir.sync_block_set {ssbuffer.block_id = 40 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.sync_block_set {ssbuffer.block_id = 40 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        %18:2 = scf.for %arg16 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
          %19 = arith.index_cast %arg18 {ssbuffer.block_id = 0 : i32} : i32 to index
          %20 = arith.muli %19, %c128 {ssbuffer.block_id = 0 : i32} : index
          %21 = arith.addi %20, %11 {ssbuffer.block_id = 0 : i32} : index
          %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%21], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 0 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
          %alloc_8 = memref.alloc() {ssbuffer.block_id = 0 : i32} : memref<128x128xf16>
          memref.copy %reinterpret_cast_7, %alloc_8 {ssbuffer.block_id = 0 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
          %22 = bufferization.to_tensor %alloc_8 restrict writable {ssbuffer.block_id = 0 : i32} : memref<128x128xf16>
          %23 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf16>
          %transposed = linalg.transpose ins(%22 : tensor<128x128xf16>) outs(%23 : tensor<128x128xf16>) permutation = [1, 0]  {ssbuffer.block_id = 0 : i32}
          %24 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 0 : i32} ins(%17, %transposed : tensor<128x128xf16>, tensor<128x128xf16>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32} ins(%24 : tensor<128x128xf32>) outs(%alloc_9 : memref<128x128xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.block_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %25 = arith.addi %arg17, %c128_i32 {ssbuffer.block_id = 4 : i32} : i32
          %26 = arith.addi %arg18, %c128_i32 {ssbuffer.block_id = 4 : i32} : i32
          hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 2
          %27 = hivm.hir.convert_layout %alloc_10 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32} : (memref<16x8x16x8xf32, #hivm.address_space<cbuf>>) -> memref<128x128xf32, #hivm.address_space<cbuf>>
          %memspacecast = memref.memory_space_cast %27 {ssbuffer.block_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<cbuf>> to memref<128x128xf32>
          %28 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 1 : i32} : memref<128x128xf32>
          %29 = hivm.hir.convert_layout %alloc_11 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
          %memspacecast_12 = memref.memory_space_cast %29 {ssbuffer.block_id = 1 : i32} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
          %30 = bufferization.to_tensor %memspacecast_12 restrict writable {ssbuffer.block_id = 1 : i32} : memref<128x128xf16>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
          %31 = arith.index_cast %arg17 {ssbuffer.block_id = 1 : i32} : i32 to index
          %32 = arith.muli %31, %c128 {ssbuffer.block_id = 1 : i32} : index
          %33 = arith.addi %32, %11 {ssbuffer.block_id = 1 : i32} : index
          %reinterpret_cast_13 = memref.reinterpret_cast %arg4 to offset: [%33], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 1 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
          %alloc_14 = memref.alloc() {ssbuffer.block_id = 1 : i32} : memref<128x128xf16>
          memref.copy %reinterpret_cast_13, %alloc_14 {ssbuffer.block_id = 1 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
          %34 = bufferization.to_tensor %alloc_14 restrict writable {ssbuffer.block_id = 1 : i32} : memref<128x128xf16>
          %35 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 1 : i32} ins(%30, %34 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%28 : tensor<128x128xf32>) -> tensor<128x128xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
          hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32} ins(%35 : tensor<128x128xf32>) outs(%alloc_15 : memref<128x128xf32, #hivm.address_space<ub>>)
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

