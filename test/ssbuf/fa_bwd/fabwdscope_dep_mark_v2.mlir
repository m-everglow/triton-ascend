module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @_attn_bwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: f32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 7 : i32} dense<[8, 8, 16, 16]> : tensor<4xi64>
    %cst_0 = arith.constant {ssbuffer.block_id = 7 : i32} dense<[128, 8, 16]> : tensor<3xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 6 : i32} 0.000000e+00 : f32
    %c128_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 128 : i32
    %c524288_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 524288 : i32
    %c65536_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 65536 : i32
    %c1024_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 1024 : i32
    %c8_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 8 : i32
    %c8192_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 8192 : i32
    %c28_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 28 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 0 : i32
    %c1_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 1 : i32
    %c64 = arith.constant {ssbuffer.block_id = 12 : i32} 64 : index
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 12 : i32} : tensor<128x128xf32>
      %1 = linalg.fill {ssbuffer.block_id = 12 : i32} ins(%arg11 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %alloc = memref.alloc() {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 6 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 6 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
      %alloc_2 = memref.alloc() {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 7 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 7 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
      scf.for %arg18 = %arg15 to %c8192_i32 step %c28_i32  : i32 {
        %2 = arith.divsi %arg18, %c8_i32 {ssbuffer.block_id = 10 : i32} : i32
        %3 = arith.muli %2, %c1024_i32 {ssbuffer.block_id = 10 : i32} : i32
        %4 = arith.remsi %2, %c8_i32 {ssbuffer.block_id = 10 : i32} : i32
        %5 = arith.muli %4, %c65536_i32 {ssbuffer.block_id = 10 : i32} : i32
        %6 = arith.divsi %2, %c8_i32 {ssbuffer.block_id = 10 : i32} : i32
        %7 = arith.muli %6, %c524288_i32 {ssbuffer.block_id = 10 : i32} : i32
        %8 = arith.addi %5, %7 {ssbuffer.block_id = 10 : i32} : i32
        %9 = arith.index_cast %8 {ssbuffer.block_id = 10 : i32} : i32 to index
        %10 = arith.index_cast %3 {ssbuffer.block_id = 10 : i32} : i32 to index
        %alloc_3 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_4 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_6 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %alloc_7 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 4 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 4 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
        %alloc_8 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
        %alloc_9 = memref.alloc() : memref<128x128xf16, #hivm.address_space<ub>>
        %memspacecast = memref.memory_space_cast %alloc_9 : memref<128x128xf16, #hivm.address_space<ub>> to memref<128x128xf16>
        %11 = bufferization.to_tensor %memspacecast writable : memref<128x128xf16>
        %alloc_10 = memref.alloc() : memref<128x128xf16, #hivm.address_space<ub>>
        %memspacecast_11 = memref.memory_space_cast %alloc_10 : memref<128x128xf16, #hivm.address_space<ub>> to memref<128x128xf16>
        %12 = bufferization.to_tensor %memspacecast_11 writable : memref<128x128xf16>
        %alloc_12 = memref.alloc() : memref<1xi64, #hivm.address_space<ub>>
        %alloc_13 = memref.alloc() : memref<1xi64, #hivm.address_space<ub>>
        %alloc_14 = memref.alloc() : memref<1xi64, #hivm.address_space<ub>>
        %alloc_15 = memref.alloc() : memref<1xi64, #hivm.address_space<ub>>
        scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          hivm.hir.sync_block_wait {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
          %memspacecast_19 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %23 = bufferization.to_tensor %memspacecast_19 restrict writable {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x128xf32>
          %24 = arith.muli %arg19, %c128_i32 {ssbuffer.block_id = 7 : i32} : i32
          %25 = arith.index_cast %24 {ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = 2 : i32} : i32 to index
          %26 = arith.addi %10, %25 {ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = 1 : i32} : index
          %reinterpret_cast_20 = memref.reinterpret_cast %arg9 to offset: [%26], sizes: [128], strides: [1] {ssbuffer.block_id = 7 : i32} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
          %alloc_21 = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<128xf32>
          memref.copy %reinterpret_cast_20, %alloc_21 {ssbuffer.block_id = 7 : i32} : memref<128xf32, strided<[1], offset: ?>> to memref<128xf32>
          %27 = bufferization.to_tensor %alloc_21 restrict writable {ssbuffer.block_id = 7 : i32} : memref<128xf32>
          %28 = arith.mulf %23, %1 {ssbuffer.block_id = 7 : i32} : tensor<128x128xf32>
          %broadcasted = linalg.broadcast ins(%27 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 7 : i32}
          %29 = arith.subf %28, %broadcasted {ssbuffer.block_id = 7 : i32} : tensor<128x128xf32>
          %30 = math.exp %29 {ssbuffer.block_id = 7 : i32} : tensor<128x128xf32>
          %31 = arith.truncf %30 {ssbuffer.block_id = 7 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
          %c0_i32_22 = arith.constant {ssbuffer.block_id = 7 : i32} 0 : i32
          %c2_i32 = arith.constant 2 : i32
          %32 = arith.remsi %arg19, %c2_i32 {ssbuffer.block_id = 7 : i32} : i32
          %33 = arith.cmpi eq, %32, %c0_i32_22 {ssbuffer.block_id = 7 : i32} : i32
          scf.if %33 {
            bufferization.materialize_in_destination %31 in restrict writable %memspacecast {ssbuffer.block_id = 7 : i32} : (tensor<128x128xf16>, memref<128x128xf16>) -> ()
          } else {
            bufferization.materialize_in_destination %31 in restrict writable %memspacecast_11 {ssbuffer.block_id = 7 : i32} : (tensor<128x128xf16>, memref<128x128xf16>) -> ()
          } {ssbuffer.block_id = 7 : i32}
          %reshape = tensor.reshape %31(%cst_0) {ssbuffer.block_id = 7 : i32} : (tensor<128x128xf16>, tensor<3xi64>) -> tensor<128x8x16xf16>
          annotation.mark %reshape {ssbuffer.block_id = 7 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<128x8x16xf16>
          %34 = tensor.empty() {ssbuffer.block_id = 7 : i32} : tensor<8x128x16xf16>
          %transposed = linalg.transpose ins(%reshape : tensor<128x8x16xf16>) outs(%34 : tensor<8x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 7 : i32}
          %reshape_23 = tensor.reshape %transposed(%cst) {ssbuffer.block_id = 7 : i32} : (tensor<8x128x16xf16>, tensor<4xi64>) -> tensor<8x8x16x16xf16>
          annotation.mark %reshape_23 {ssbuffer.block_id = 7 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<8x8x16x16xf16>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
          hivm.hir.copy ins(%reshape_23 : tensor<8x8x16x16xf16>) outs(%alloc_4 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 1 : i32}
          hivm.hir.sync_block_set {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          hivm.hir.sync_block_set {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
          hivm.hir.sync_block_wait {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 5
          %memspacecast_24 = memref.memory_space_cast %alloc_7 {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 4 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %35 = bufferization.to_tensor %memspacecast_24 restrict writable {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 4 : i32} : memref<128x128xf32>
          %reinterpret_cast_25 = memref.reinterpret_cast %arg10 to offset: [%26], sizes: [128], strides: [1] {ssbuffer.block_id = 8 : i32, ssbuffer.dep_mark = 1 : i32} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
          %alloc_26 = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<128xf32>
          memref.copy %reinterpret_cast_25, %alloc_26 {ssbuffer.block_id = 8 : i32} : memref<128xf32, strided<[1], offset: ?>> to memref<128xf32>
          %36 = bufferization.to_tensor %alloc_26 restrict writable {ssbuffer.block_id = 8 : i32} : memref<128xf32>
          %broadcasted_27 = linalg.broadcast ins(%36 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 8 : i32}
          %37 = arith.subf %35, %broadcasted_27 {ssbuffer.block_id = 8 : i32} : tensor<128x128xf32>
          %c2_i32_28 = arith.constant {ssbuffer.block_id = 8 : i32} 2 : i32
          %c1_i32_29 = arith.constant {ssbuffer.block_id = 8 : i32} 1 : i32
          %38 = arith.addi %arg19, %c1_i32_29 {ssbuffer.block_id = 8 : i32} : i32
          %39 = arith.remsi %38, %c2_i32_28 {ssbuffer.block_id = 8 : i32} : i32
          %c0_i32_30 = arith.constant {ssbuffer.block_id = 8 : i32} 0 : i32
          %40 = arith.cmpi eq, %39, %c0_i32_30 {ssbuffer.block_id = 8 : i32} : i32
          %41 = scf.if %40 -> (tensor<128x128xf16>) {
            %51 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 8 : i32} : memref<128x128xf16>
            scf.yield %51 : tensor<128x128xf16>
          } else {
            %51 = bufferization.to_tensor %memspacecast_11 restrict writable {ssbuffer.block_id = 8 : i32} : memref<128x128xf16>
            scf.yield %51 : tensor<128x128xf16>
          } {ssbuffer.block_id = 8 : i32}
          %42 = arith.extf %41 {ssbuffer.block_id = 8 : i32} : tensor<128x128xf16> to tensor<128x128xf32>
          %43 = arith.mulf %42, %37 {ssbuffer.block_id = 8 : i32} : tensor<128x128xf32>
          %44 = arith.mulf %43, %1 {ssbuffer.block_id = 8 : i32} : tensor<128x128xf32>
          %45 = arith.truncf %44 {ssbuffer.block_id = 8 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
          %reshape_31 = tensor.reshape %45(%cst_0) {ssbuffer.block_id = 8 : i32} : (tensor<128x128xf16>, tensor<3xi64>) -> tensor<128x8x16xf16>
          annotation.mark %reshape_31 {ssbuffer.block_id = 8 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<128x8x16xf16>
          %46 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<8x128x16xf16>
          %transposed_32 = linalg.transpose ins(%reshape_31 : tensor<128x8x16xf16>) outs(%46 : tensor<8x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 8 : i32}
          %reshape_33 = tensor.reshape %transposed_32(%cst) {ssbuffer.block_id = 8 : i32} : (tensor<8x128x16xf16>, tensor<4xi64>) -> tensor<8x8x16x16xf16>
          annotation.mark %reshape_33 {ssbuffer.block_id = 8 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<8x8x16x16xf16>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          hivm.hir.copy ins(%reshape_33 : tensor<8x8x16x16xf16>) outs(%alloc_3 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}
          hivm.hir.sync_block_set {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
          hivm.hir.sync_block_wait {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 3
          hivm.hir.copy ins(%reshape_33 : tensor<8x8x16x16xf16>) outs(%alloc_5 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32}
          hivm.hir.sync_block_set {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
          hivm.hir.sync_block_set {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
          hivm.hir.sync_block_wait {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
          %memspacecast_34 = memref.memory_space_cast %alloc_8 {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x64xf32, #hivm.address_space<ub>> to memref<128x64xf32>
          %47 = bufferization.to_tensor %memspacecast_34 restrict writable {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x64xf32>
          %48 = arith.muli %25, %c64 {ssbuffer.block_id = 9 : i32, ssbuffer.dep_mark = 2 : i32} : index
          %49 = arith.addi %9, %48 {ssbuffer.block_id = 9 : i32} : index
          %reinterpret_cast_35 = memref.reinterpret_cast %arg6 to offset: [%49], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 9 : i32} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
          %50 = arith.truncf %47 {ssbuffer.block_id = 9 : i32} : tensor<128x64xf32> to tensor<128x64xf16>
          hivm.hir.store ins(%50 : tensor<128x64xf16>) outs(%reinterpret_cast_35 : memref<128x64xf16, strided<[64, 1], offset: ?>>) {ssbuffer.block_id = 9 : i32} atomic = <add>
          hivm.hir.sync_block_set {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
        } {ssbuffer.block_id = 13 : i32, ssbuffer.main_loop = 0 : i64}
        hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 3
        hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 8
        %memspacecast_16 = memref.memory_space_cast %alloc_2 {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 7 : i32} : memref<128x64xf32, #hivm.address_space<ub>> to memref<128x64xf32>
        %13 = bufferization.to_tensor %memspacecast_16 restrict writable {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 7 : i32} : memref<128x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 7
        %memspacecast_17 = memref.memory_space_cast %alloc {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 6 : i32} : memref<128x64xf32, #hivm.address_space<ub>> to memref<128x64xf32>
        %14 = bufferization.to_tensor %memspacecast_17 restrict writable {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 6 : i32} : memref<128x64xf32>
        %15 = arith.muli %2, %c8_i32 {ssbuffer.block_id = 11 : i32} : i32
        %16 = arith.subi %arg18, %15 {ssbuffer.block_id = 11 : i32} : i32
        %17 = arith.muli %16, %c128_i32 {ssbuffer.block_id = 11 : i32} : i32
        %18 = arith.index_cast %17 {ssbuffer.block_id = 11 : i32} : i32 to index
        %19 = arith.muli %18, %c64 {ssbuffer.block_id = 11 : i32} : index
        %20 = arith.addi %9, %19 {ssbuffer.block_id = 11 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%20], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 11 : i32} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
        %21 = arith.truncf %14 {ssbuffer.block_id = 11 : i32} : tensor<128x64xf32> to tensor<128x64xf16>
        bufferization.materialize_in_destination %21 in writable %reinterpret_cast {ssbuffer.block_id = 11 : i32} : (tensor<128x64xf16>, memref<128x64xf16, strided<[64, 1], offset: ?>>) -> ()
        %reinterpret_cast_18 = memref.reinterpret_cast %arg8 to offset: [%20], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 11 : i32} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
        %22 = arith.truncf %13 {ssbuffer.block_id = 11 : i32} : tensor<128x64xf32> to tensor<128x64xf16>
        bufferization.materialize_in_destination %22 in writable %reinterpret_cast_18 {ssbuffer.block_id = 11 : i32} : (tensor<128x64xf16>, memref<128x64xf16, strided<[64, 1], offset: ?>>) -> ()
        hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
        hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
      } {ssbuffer.block_id = 14 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 6 : i32} : tensor<128x64xf32>
      %1 = linalg.fill {ssbuffer.block_id = 6 : i32} ins(%cst_1 : f32) outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %2 = tensor.empty() {ssbuffer.block_id = 6 : i32} : tensor<128x128xf32>
      %3 = linalg.fill {ssbuffer.block_id = 6 : i32} ins(%cst_1 : f32) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %alloc = memref.alloc() {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 6 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 6 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      %alloc_2 = memref.alloc() {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 7 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 7 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      scf.for %arg18 = %arg15 to %c8192_i32 step %c28_i32  : i32 {
        %4 = arith.divsi %arg18, %c8_i32 {ssbuffer.block_id = 5 : i32} : i32
        %5 = arith.muli %4, %c8_i32 {ssbuffer.block_id = 5 : i32} : i32
        %6 = arith.subi %arg18, %5 {ssbuffer.block_id = 5 : i32} : i32
        %7 = arith.remsi %4, %c8_i32 {ssbuffer.block_id = 5 : i32} : i32
        %8 = arith.muli %7, %c65536_i32 {ssbuffer.block_id = 5 : i32} : i32
        %9 = arith.divsi %4, %c8_i32 {ssbuffer.block_id = 5 : i32} : i32
        %10 = arith.muli %9, %c524288_i32 {ssbuffer.block_id = 5 : i32} : i32
        %11 = arith.addi %8, %10 {ssbuffer.block_id = 5 : i32} : i32
        %12 = arith.index_cast %11 {ssbuffer.block_id = 5 : i32} : i32 to index
        %13 = arith.muli %6, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
        %14 = arith.index_cast %13 {ssbuffer.block_id = 5 : i32} : i32 to index
        %15 = arith.muli %14, %c64 {ssbuffer.block_id = 5 : i32} : index
        %16 = arith.addi %12, %15 {ssbuffer.block_id = 5 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%16], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 5 : i32} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
        %alloc_3 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<128x64xf16>
        memref.copy %reinterpret_cast, %alloc_3 {ssbuffer.block_id = 5 : i32} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<128x64xf16>
        %17 = bufferization.to_tensor %alloc_3 restrict writable {ssbuffer.block_id = 5 : i32} : memref<128x64xf16>
        %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%16], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 5 : i32} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<128x64xf16>
        memref.copy %reinterpret_cast_4, %alloc_5 {ssbuffer.block_id = 5 : i32} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<128x64xf16>
        %18 = bufferization.to_tensor %alloc_5 restrict writable {ssbuffer.block_id = 5 : i32} : memref<128x64xf16>
        %19 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<64x128xf16>
        %transposed = linalg.transpose ins(%17 : tensor<128x64xf16>) outs(%19 : tensor<64x128xf16>) permutation = [1, 0]  {ssbuffer.block_id = 5 : i32}
        %transposed_6 = linalg.transpose ins(%18 : tensor<128x64xf16>) outs(%19 : tensor<64x128xf16>) permutation = [1, 0]  {ssbuffer.block_id = 5 : i32}
        %alloc_7 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        %alloc_8 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        %alloc_9 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 3
        %alloc_10 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_10 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        %alloc_11 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 4 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_11 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 4 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        %alloc_12 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_12 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
        %20:2 = scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg20 = %1, %arg21 = %1) -> (tensor<128x64xf32>, tensor<128x64xf32>)  : i32 {
          %21 = arith.muli %arg19, %c128_i32 {ssbuffer.block_id = 0 : i32} : i32
          %22 = arith.index_cast %21 {ssbuffer.block_id = 0 : i32} : i32 to index
          %23 = arith.muli %22, %c64 {ssbuffer.block_id = 0 : i32} : index
          %24 = arith.addi %12, %23 {ssbuffer.block_id = 0 : i32} : index
          %reinterpret_cast_13 = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 0 : i32} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
          %alloc_14 = memref.alloc() {ssbuffer.block_id = 0 : i32} : memref<128x64xf16>
          memref.copy %reinterpret_cast_13, %alloc_14 {ssbuffer.block_id = 0 : i32} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<128x64xf16>
          %25 = bufferization.to_tensor %alloc_14 restrict writable {ssbuffer.block_id = 0 : i32} : memref<128x64xf16>
          %26 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 0 : i32} ins(%25, %transposed : tensor<128x64xf16>, tensor<64x128xf16>) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 3 : i32} ins(%26 : tensor<128x128xf32>) outs(%alloc_10 : memref<128x128xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 4
          hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          %27 = hivm.hir.convert_layout %alloc_8 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
          %memspacecast = memref.memory_space_cast %27 {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
          %28 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf16>
          %reinterpret_cast_15 = memref.reinterpret_cast %arg5 to offset: [%24], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 1 : i32} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
          %alloc_16 = memref.alloc() {ssbuffer.block_id = 1 : i32} : memref<128x64xf16>
          memref.copy %reinterpret_cast_15, %alloc_16 {ssbuffer.block_id = 1 : i32} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<128x64xf16>
          %29 = bufferization.to_tensor %alloc_16 restrict writable {ssbuffer.block_id = 1 : i32} : memref<128x64xf16>
          %30 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128x128xf16>
          %transposed_17 = linalg.transpose ins(%28 : tensor<128x128xf16>) outs(%30 : tensor<128x128xf16>) permutation = [1, 0]  {ssbuffer.block_id = 1 : i32}
          %31 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 1 : i32} ins(%transposed_17, %29 : tensor<128x128xf16>, tensor<128x64xf16>) outs(%arg21 : tensor<128x64xf32>) -> tensor<128x64xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
          %32 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32} ins(%29, %transposed_6 : tensor<128x64xf16>, tensor<64x128xf16>) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 4 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 4 : i32} ins(%32 : tensor<128x128xf32>) outs(%alloc_11 : memref<128x128xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 4 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 5
          hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
          %33 = hivm.hir.convert_layout %alloc_9 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 2 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
          %memspacecast_18 = memref.memory_space_cast %33 {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
          %34 = bufferization.to_tensor %memspacecast_18 restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf16>
          %transposed_19 = linalg.transpose ins(%34 : tensor<128x128xf16>) outs(%30 : tensor<128x128xf16>) permutation = [1, 0]  {ssbuffer.block_id = 3 : i32}
          %35 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32} ins(%transposed_19, %25 : tensor<128x128xf16>, tensor<128x64xf16>) outs(%arg20 : tensor<128x64xf32>) -> tensor<128x64xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 3
          hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
          %36 = hivm.hir.convert_layout %alloc_7 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
          %memspacecast_20 = memref.memory_space_cast %36 {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
          %37 = bufferization.to_tensor %memspacecast_20 restrict writable {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16>
          %38 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 4 : i32} ins(%37, %17 : tensor<128x128xf16>, tensor<128x64xf16>) outs(%1 : tensor<128x64xf32>) -> tensor<128x64xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 5 : i32} ins(%38 : tensor<128x64xf32>) outs(%alloc_12 : memref<128x64xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
          scf.yield %35, %31 : tensor<128x64xf32>, tensor<128x64xf32>
        } {ssbuffer.block_id = 13 : i32, ssbuffer.main_loop = 0 : i64}
        hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
        hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 4 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
        hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
        hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 6 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 7
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 6 : i32} ins(%20#0 : tensor<128x64xf32>) outs(%alloc : memref<128x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 6 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 7
        hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 7 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 8
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 7 : i32} ins(%20#1 : tensor<128x64xf32>) outs(%alloc_2 : memref<128x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 7 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 8
      } {ssbuffer.block_id = 14 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 7 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 8
      hivm.hir.sync_block_wait {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 6 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 7
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}

