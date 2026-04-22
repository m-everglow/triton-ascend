module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 1 : i32} dense<0.000000e+00> : tensor<128x128xf32>
    %cst_0 = arith.constant {ssbuffer.block_id = 6 : i32} dense<[8, 8, 16, 16]> : tensor<4xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 6 : i32} dense<[128, 8, 16]> : tensor<3xi64>
    %cst_2 = arith.constant {ssbuffer.block_id = 11 : i32} 0.000000e+00 : f32
    %c28_i32 = arith.constant {ssbuffer.block_id = 11 : i32} 28 : i32
    %c65536_i32 = arith.constant {ssbuffer.block_id = 11 : i32} 65536 : i32
    %c8192_i32 = arith.constant {ssbuffer.block_id = 11 : i32} 8192 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 11 : i32} 128 : i32
    %c1048576_i64 = arith.constant {ssbuffer.block_id = 11 : i32} 1048576 : i64
    %c8388608_i64 = arith.constant {ssbuffer.block_id = 11 : i32} 8388608 : i64
    %c8_i32 = arith.constant {ssbuffer.block_id = 11 : i32} 8 : i32
    %c64_i32 = arith.constant {ssbuffer.block_id = 11 : i32} 64 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 11 : i32} 0 : i32
    %cst_3 = arith.constant {ssbuffer.block_id = 11 : i32} 5.000000e-01 : f32
    %cst_4 = arith.constant {ssbuffer.block_id = 11 : i32} 0xFF800000 : f32
    %cst_5 = arith.constant {ssbuffer.block_id = 11 : i32} 1.000000e+00 : f32
    %c128 = arith.constant {ssbuffer.block_id = 11 : i32} 128 : index
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<128x128xf32>
      %1 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<128x128xf32>
      %2 = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst_2 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %3 = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst_3 : f32) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %4 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<128xf32>
      %5 = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst_4 : f32) outs(%4 : tensor<128xf32>) -> tensor<128xf32>
      %6 = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst_5 : f32) outs(%4 : tensor<128xf32>) -> tensor<128xf32>
      scf.for %arg16 = %arg13 to %c65536_i32 step %c28_i32  : i32 {
        %7 = arith.divsi %arg16, %c64_i32 {ssbuffer.block_id = 9 : i32} : i32
        %8 = arith.remsi %arg16, %c64_i32 {ssbuffer.block_id = 9 : i32} : i32
        %9 = arith.divsi %7, %c8_i32 {ssbuffer.block_id = 9 : i32} : i32
        %10 = arith.remsi %7, %c8_i32 {ssbuffer.block_id = 9 : i32} : i32
        %11 = arith.extsi %9 {ssbuffer.block_id = 9 : i32} : i32 to i64
        %12 = arith.muli %11, %c8388608_i64 {ssbuffer.block_id = 9 : i32} : i64
        %13 = arith.extsi %10 {ssbuffer.block_id = 9 : i32} : i32 to i64
        %14 = arith.muli %13, %c1048576_i64 {ssbuffer.block_id = 9 : i32} : i64
        %15 = arith.addi %12, %14 {ssbuffer.block_id = 9 : i32} : i64
        %16 = arith.index_cast %15 {ssbuffer.block_id = 9 : i32} : i64 to index
        %17 = arith.muli %8, %c128_i32 {ssbuffer.block_id = 9 : i32} : i32
        %18 = arith.maxsi %17, %c0_i32 {ssbuffer.block_id = 9 : i32} : i32
        %19 = arith.index_cast %18 {ssbuffer.block_id = 9 : i32} : i32 to index
        %20 = arith.muli %19, %c128 {ssbuffer.block_id = 9 : i32} : index
        %21 = arith.addi %20, %16 {ssbuffer.block_id = 9 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%21], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 9 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
        %22 = arith.muli %7, %c8192_i32 {ssbuffer.block_id = 9 : i32} : i32
        %23 = arith.index_cast %22 {ssbuffer.block_id = 9 : i32} : i32 to index
        %24 = arith.index_cast %17 {ssbuffer.block_id = 9 : i32} : i32 to index
        %25 = arith.addi %23, %24 {ssbuffer.block_id = 9 : i32} : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%25], sizes: [128], strides: [1] {ssbuffer.block_id = 9 : i32} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
        %alloc = memref.alloc() {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2} : memref<128x128xf32, #hivm.address_space<ub>>
        %alloc_9 = memref.alloc() {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 1} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 1} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_12 = memref.alloc() {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 3} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_12 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 3} : memref<128x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
        hivm.hir.sync_block_set {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 3}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
        %26:3 = scf.for %arg17 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg18 = %6, %arg19 = %2, %arg20 = %5) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>)  : i32 {
          %31 = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_2 : f32) outs(%4 : tensor<128xf32>) -> tensor<128xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
          %memspacecast = memref.memory_space_cast %alloc {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %32 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2} : memref<128x128xf32>
          %33 = arith.mulf %32, %3 {ssbuffer.block_id = 6 : i32} : tensor<128x128xf32>
          %reduced = linalg.reduce ins(%33 : tensor<128x128xf32>) outs(%5 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 6 : i32}
            (%in: f32, %init: f32) {
              %46 = arith.maximumf %in, %init : f32
              linalg.yield %46 : f32
            }
          %34 = arith.maximumf %arg20, %reduced {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
          %broadcasted_7 = linalg.broadcast ins(%34 : tensor<128xf32>) outs(%1 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 6 : i32}
          %35 = arith.subf %33, %broadcasted_7 {ssbuffer.block_id = 6 : i32} : tensor<128x128xf32>
          %36 = math.exp %35 {ssbuffer.block_id = 6 : i32} : tensor<128x128xf32>
          %37 = arith.truncf %36 {ssbuffer.block_id = 6 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
          %reshape = tensor.reshape %37(%cst_1) {ssbuffer.block_id = 6 : i32} : (tensor<128x128xf16>, tensor<3xi64>) -> tensor<128x8x16xf16>
          annotation.mark %reshape {ssbuffer.block_id = 6 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<128x8x16xf16>
          %38 = tensor.empty() {ssbuffer.block_id = 6 : i32} : tensor<8x128x16xf16>
          %transposed = linalg.transpose ins(%reshape : tensor<128x8x16xf16>) outs(%38 : tensor<8x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 6 : i32}
          %reshape_8 = tensor.reshape %transposed(%cst_0) {ssbuffer.block_id = 6 : i32} : (tensor<8x128x16xf16>, tensor<4xi64>) -> tensor<8x8x16x16xf16>
          annotation.mark %reshape_8 {ssbuffer.block_id = 6 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<8x8x16x16xf16>
          hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
          hivm.hir.sync_block_wait {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 1}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          hivm.hir.copy ins(%reshape_8 : tensor<8x8x16x16xf16>) outs(%alloc_9 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 1}
          hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 1}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
          %reduced_10 = linalg.reduce ins(%36 : tensor<128x128xf32>) outs(%31 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 7 : i32}
            (%in: f32, %init: f32) {
              %46 = arith.addf %in, %init : f32
              linalg.yield %46 : f32
            }
          %39 = arith.subf %arg20, %34 {ssbuffer.block_id = 7 : i32} : tensor<128xf32>
          %40 = math.exp %39 {ssbuffer.block_id = 7 : i32} : tensor<128xf32>
          %41 = arith.mulf %arg18, %40 {ssbuffer.block_id = 7 : i32} : tensor<128xf32>
          %42 = arith.addf %41, %reduced_10 {ssbuffer.block_id = 7 : i32} : tensor<128xf32>
          %broadcasted_11 = linalg.broadcast ins(%40 : tensor<128xf32>) outs(%1 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 7 : i32}
          %43 = arith.mulf %arg19, %broadcasted_11 {ssbuffer.block_id = 7 : i32} : tensor<128x128xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 3}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %memspacecast_13 = memref.memory_space_cast %alloc_12 {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 3} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %44 = bufferization.to_tensor %memspacecast_13 restrict writable {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 3} : memref<128x128xf32>
          %45 = arith.addf %44, %43 {ssbuffer.block_id = 8 : i32} : tensor<128x128xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 3}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
          scf.yield %42, %45, %34 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>
        } {ssbuffer.block_id = 40 : i32, ssbuffer.main_loop = 1 : i32}
        hivm.hir.sync_block_wait {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 1}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        %27 = math.log %26#0 {ssbuffer.block_id = 10 : i32} : tensor<128xf32>
        %28 = arith.addf %26#2, %27 {ssbuffer.block_id = 10 : i32} : tensor<128xf32>
        %broadcasted = linalg.broadcast ins(%26#0 : tensor<128xf32>) outs(%1 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 10 : i32}
        %29 = arith.divf %26#1, %broadcasted {ssbuffer.block_id = 10 : i32} : tensor<128x128xf32>
        bufferization.materialize_in_destination %28 in writable %reinterpret_cast_6 {ssbuffer.block_id = 10 : i32} : (tensor<128xf32>, memref<128xf32, strided<[1], offset: ?>>) -> ()
        %30 = arith.truncf %29 {ssbuffer.block_id = 10 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
        bufferization.materialize_in_destination %30 in writable %reinterpret_cast {ssbuffer.block_id = 10 : i32} : (tensor<128x128xf16>, memref<128x128xf16, strided<[128, 1], offset: ?>>) -> ()
      } {ssbuffer.block_id = 41 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<128x128xf32>
      %1 = linalg.fill {ssbuffer.block_id = 4 : i32} ins(%cst_2 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.for %arg16 = %arg13 to %c65536_i32 step %c28_i32  : i32 {
        %2 = arith.divsi %arg16, %c64_i32 {ssbuffer.block_id = 3 : i32} : i32
        %3 = arith.remsi %arg16, %c64_i32 {ssbuffer.block_id = 3 : i32} : i32
        %4 = arith.divsi %2, %c8_i32 {ssbuffer.block_id = 3 : i32} : i32
        %5 = arith.remsi %2, %c8_i32 {ssbuffer.block_id = 3 : i32} : i32
        %6 = arith.muli %5, %arg9 {ssbuffer.block_id = 3 : i32} : i32
        %7 = arith.divsi %6, %c8_i32 {ssbuffer.block_id = 3 : i32} : i32
        %8 = arith.extsi %4 {ssbuffer.block_id = 3 : i32} : i32 to i64
        %9 = arith.muli %8, %c8388608_i64 {ssbuffer.block_id = 3 : i32} : i64
        %10 = arith.extsi %5 {ssbuffer.block_id = 3 : i32} : i32 to i64
        %11 = arith.muli %10, %c1048576_i64 {ssbuffer.block_id = 3 : i32} : i64
        %12 = arith.addi %9, %11 {ssbuffer.block_id = 3 : i32} : i64
        %13 = arith.extsi %7 {ssbuffer.block_id = 3 : i32} : i32 to i64
        %14 = arith.muli %13, %c1048576_i64 {ssbuffer.block_id = 3 : i32} : i64
        %15 = arith.addi %9, %14 {ssbuffer.block_id = 3 : i32} : i64
        %16 = arith.index_cast %12 {ssbuffer.block_id = 3 : i32} : i64 to index
        %17 = arith.muli %3, %c128_i32 {ssbuffer.block_id = 3 : i32} : i32
        %18 = arith.maxsi %17, %c0_i32 {ssbuffer.block_id = 3 : i32} : i32
        %19 = arith.index_cast %18 {ssbuffer.block_id = 3 : i32} : i32 to index
        %20 = arith.muli %19, %c128 {ssbuffer.block_id = 3 : i32} : index
        %21 = arith.addi %20, %16 {ssbuffer.block_id = 3 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 3 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
        %22 = arith.index_cast %15 {ssbuffer.block_id = 3 : i32} : i64 to index
        %alloc = memref.alloc() {ssbuffer.block_id = 3 : i32} : memref<128x128xf16>
        memref.copy %reinterpret_cast, %alloc {ssbuffer.block_id = 3 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
        %23 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 3 : i32} : memref<128x128xf16>
        %alloc_8 = memref.alloc() {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2} : memref<128x128xf32, #hivm.address_space<ub>>
        %alloc_9 = memref.alloc() {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 1} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 1} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_12 = memref.alloc() {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 3} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_12 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 3} : memref<128x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 1}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        %24:2 = scf.for %arg17 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg18 = %c0_i32, %arg19 = %c0_i32) -> (i32, i32)  : i32 {
          %25 = arith.index_cast %arg19 {ssbuffer.block_id = 0 : i32} : i32 to index
          %26 = arith.muli %25, %c128 {ssbuffer.block_id = 0 : i32} : index
          %27 = arith.addi %26, %22 {ssbuffer.block_id = 0 : i32} : index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 0 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
          %alloc_7 = memref.alloc() {ssbuffer.block_id = 0 : i32} : memref<128x128xf16>
          memref.copy %reinterpret_cast_6, %alloc_7 {ssbuffer.block_id = 0 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
          %28 = bufferization.to_tensor %alloc_7 restrict writable {ssbuffer.block_id = 0 : i32} : memref<128x128xf16>
          %29 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf16>
          %transposed = linalg.transpose ins(%28 : tensor<128x128xf16>) outs(%29 : tensor<128x128xf16>) permutation = [1, 0]  {ssbuffer.block_id = 0 : i32}
          %30 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 0 : i32} ins(%23, %transposed : tensor<128x128xf16>, tensor<128x128xf16>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 2}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 2} ins(%30 : tensor<128x128xf32>) outs(%alloc_8 : memref<128x128xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 2}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
          %31 = arith.addi %arg18, %c128_i32 {ssbuffer.block_id = 2 : i32} : i32
          %32 = arith.addi %arg19, %c128_i32 {ssbuffer.block_id = 2 : i32} : i32
          hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1}[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
          %33 = hivm.hir.convert_layout %alloc_9 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
          %memspacecast = memref.memory_space_cast %33 {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
          %34 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 1 : i32} : memref<128x128xf16>
          %35 = arith.index_cast %arg18 {ssbuffer.block_id = 1 : i32} : i32 to index
          %36 = arith.muli %35, %c128 {ssbuffer.block_id = 1 : i32} : index
          %37 = arith.addi %36, %22 {ssbuffer.block_id = 1 : i32} : index
          %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [%37], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 1 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
          %alloc_11 = memref.alloc() {ssbuffer.block_id = 1 : i32} : memref<128x128xf16>
          memref.copy %reinterpret_cast_10, %alloc_11 {ssbuffer.block_id = 1 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
          %38 = bufferization.to_tensor %alloc_11 restrict writable {ssbuffer.block_id = 1 : i32} : memref<128x128xf16>
          %39 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 1 : i32} ins(%34, %38 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%cst : tensor<128x128xf32>) -> tensor<128x128xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 3}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 3} ins(%39 : tensor<128x128xf32>) outs(%alloc_12 : memref<128x128xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 3}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
          scf.yield %31, %32 : i32, i32
        } {ssbuffer.block_id = 40 : i32, ssbuffer.main_loop = 1 : i32}
        hivm.hir.sync_block_wait {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 3}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
        hivm.hir.sync_block_wait {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
      } {ssbuffer.block_id = 41 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}

