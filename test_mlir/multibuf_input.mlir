module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} 0 : i32
    %c128_i32 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} 128 : i32
    %c8_i32 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} 8 : i32
    %c8388608_i64 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} 8388608 : i64
    %c1048576_i64 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} 1048576 : i64
    %c64_i32 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} 64 : i32
    %c8192_i64 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} 8192 : i64
    %c128_i64 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} 128 : i64
    %c1_i64 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} 1 : i64
    %c131072_i32 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} 131072 : i32
    %c28_i32 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} 28 : i32
    %cst = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} dense<0.000000e+00> : tensor<64x128xf32>
    %cst_0 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} dense<5.000000e-01> : tensor<64x64xf32>
    %cst_1 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} dense<0.000000e+00> : tensor<64x64xf32>
    %c8192_i32 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} 8192 : i32
    %cst_2 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} dense<0xFF800000> : tensor<64xf32>
    %cst_3 = arith.constant {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} dense<1.000000e+00> : tensor<64xf32>
    %alloc = memref.alloc() {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    %alloc_4 = memref.alloc() {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
    %alloc_5 = memref.alloc() {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    %0 = tt.get_program_id x {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32} : i32
    scope.scope : () -> () {
      hivm.hir.sync_block_set {ssbuffer.end_nesting_depth = 1 : i32, ssbuffer.nesting_depth = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
      hivm.hir.sync_block_set {ssbuffer.end_nesting_depth = 1 : i32, ssbuffer.nesting_depth = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
      scf.for %arg7 = %0 to %c131072_i32 step %c28_i32  : i32 {
        %1 = arith.divsi %arg7, %c128_i32 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i32
        %2 = arith.remsi %arg7, %c128_i32 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i32
        %3 = arith.divsi %1, %c8_i32 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i32
        %4 = arith.remsi %1, %c8_i32 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i32
        %5 = arith.extsi %3 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i32 to i64
        %6 = arith.muli %5, %c8388608_i64 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i64
        %7 = arith.extsi %4 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i32 to i64
        %8 = arith.muli %7, %c1048576_i64 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i64
        %9 = arith.addi %6, %8 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i64
        %10 = arith.muli %2, %c64_i32 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i32
        %11 = tt.addptr %arg2, %9 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : !tt.ptr<f16>, i64
        %12 = tt.make_tensor_ptr %11, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>, ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : <tensor<64x128xf16>>
        %13 = tt.addptr %arg1, %9 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : !tt.ptr<f16>, i64
        %14 = tt.make_tensor_ptr %13, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>, ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : <tensor<64x128xf16>>
        %15 = tt.addptr %arg5, %9 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : !tt.ptr<f16>, i64
        %16 = tt.make_tensor_ptr %15, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%10, %c0_i32] {order = array<i32: 1, 0>, ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : <tensor<64x128xf16>>
        // Buffer allocations for depVal2 (%41, type: tensor<64x128xf32>)
        %buf0_dep1 = memref.alloc() {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
        %buf1_dep1 = memref.alloc() {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
        %buf0_dep1_cast = memref.memory_space_cast %buf0_dep1 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
        %buf1_dep1_cast = memref.memory_space_cast %buf1_dep1 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
        %buf0_dep1_tensor = bufferization.to_tensor %buf0_dep1_cast restrict writable {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} : memref<64x128xf32>
        %buf1_dep1_tensor = bufferization.to_tensor %buf1_dep1_cast restrict writable {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} : memref<64x128xf32>

        // Counter initialization
        %counter = arith.constant {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} 0 : i32
        %counter1 = arith.constant {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} 0 : i32
        // Constants for double buffer control
        %c0_i32_1 = arith.constant {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} 0 : i32
        %c1_i32_1 = arith.constant {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} 1 : i32
        %c2_i32_1 = arith.constant {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} 2 : i32
        %17:9 = scf.for %arg8 = %c0_i32 to %c8192_i32 step %c64_i32 iter_args(%arg9 = %cst_3, %arg10 = %cst, %arg11 = %cst_2, %arg12 = %12, %arg13 = %14, %arg14 = %buf0_dep1_tensor, %arg15 = %buf1_dep1_tensor, %arg16 = %counter, %arg17 = %counter1) -> (tensor<64xf32>, tensor<64x128xf32>, tensor<64xf32>, !tt.ptr<tensor<64x128xf16>>, !tt.ptr<tensor<64x128xf16>>, tensor<64x128xf32>, tensor<64x128xf32>, i32, i32)  : i32 {
          hivm.hir.sync_block_wait {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          %memspacecast = memref.memory_space_cast %alloc_4 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
          %22 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : memref<64x64xf32>
          %23 = arith.mulf %22, %cst_0 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64x64xf32>
          %24 = "tt.reduce"(%23) <{axis = 1 : i32}> ({
          ^bb0(%arg214: f32, %arg215: f32):
            %46 = arith.maximumf %arg214, %arg215 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} : f32
            tt.reduce.return %46 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} : f32
          }) {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : (tensor<64x64xf32>) -> tensor<64xf32>
          %25 = arith.maximumf %arg11, %24 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64xf32>
          %26 = tt.expand_dims %25 {axis = 1 : i32, ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64xf32> -> tensor<64x1xf32>
          %27 = tt.broadcast %26 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64x1xf32> -> tensor<64x64xf32>
          %28 = arith.subf %23, %27 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64x64xf32>
          %29 = math.exp %28 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64x64xf32>
          %30 = arith.truncf %29 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64x64xf32> to tensor<64x64xf16>
          %31 = tt.reshape %30 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64x64xf16> -> tensor<4x16x4x16xf16>
          %32 = tt.trans %31 {order = array<i32: 2, 0, 1, 3>, ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<4x16x4x16xf16> -> tensor<4x4x16x16xf16>
          hivm.hir.sync_block_set {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
          hivm.hir.sync_block_wait {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 6
          %33 = bufferization.to_memref %32 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : memref<4x4x16x16xf16, #hivm.address_space<ub>>
          hivm.hir.copy ins(%33 : memref<4x4x16x16xf16, #hivm.address_space<ub>>) outs(%alloc : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1}
          hivm.hir.sync_block_set {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          %34 = "tt.reduce"(%29) <{axis = 1 : i32}> ({
          ^bb0(%arg214: f32, %arg215: f32):
            %46 = arith.addf %arg214, %arg215 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} : f32
            tt.reduce.return %46 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32} : f32
          }) {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : (tensor<64x64xf32>) -> tensor<64xf32>
          %35 = arith.subf %arg11, %25 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64xf32>
          %36 = math.exp %35 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64xf32>
          %37 = arith.mulf %arg9, %36 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64xf32>
          %38 = arith.addf %37, %34 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64xf32>
          %39 = tt.expand_dims %36 {axis = 1 : i32, ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64xf32> -> tensor<64x1xf32>
          %40 = tt.broadcast %39 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64x1xf32> -> tensor<64x128xf32>
          // ---------- Producer for depVal1: write %40 to buffer ----------
          %buf_idx_dep1 = arith.remsi %counter, %c2_i32_1 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.id=1 : i32, ssbuffer.nesting_depth = 3 : i32} : i32
          %is_buf0_dep1 = arith.cmpi eq, %buf_idx_dep1, %c0_i32_1 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.id=1 : i32, ssbuffer.nesting_depth = 3 : i32} : i32
          %mask_dep1 = tensor.splat %is_buf0_dep1 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.id=1 : i32, ssbuffer.nesting_depth = 3 : i32} : tensor<64x128xi1>
          %new_buf0_dep1 = arith.select %mask_dep1, %40, %buf0_dep1_tensor {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.id=1 : i32, ssbuffer.nesting_depth = 3 : i32} : tensor<64x128xi1>, tensor<64x128xf32>
          %new_buf1_dep1 = arith.select %mask_dep1, %buf1_dep1_tensor, %40 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.id=1 : i32, ssbuffer.nesting_depth = 3 : i32} : tensor<64x128xi1>, tensor<64x128xf32>
          %counter_after_dep1 = arith.addi %counter, %c1_i32_1 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.id=1 : i32, ssbuffer.nesting_depth = 3 : i32} : i32

          // ---------- Consumer for depVal1: select from previous buffer ----------
          %cons_buf_idx_dep1 = arith.remsi %counter1, %c2_i32_1 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.id=2 : i32, ssbuffer.nesting_depth = 3 : i32} : i32
          %is_cons_buf0_dep1 = arith.cmpi eq, %cons_buf_idx_dep1, %c0_i32_1 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.id=2 : i32, ssbuffer.nesting_depth = 3 : i32} : i32
          %cons_mask_dep1 = tensor.splat %is_cons_buf0_dep1 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.id=2 : i32, ssbuffer.nesting_depth = 3 : i32} : tensor<64x128xi1>
          %selected_dep1 = arith.select %cons_mask_dep1, %new_buf0_dep1, %new_buf1_dep1 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.id=2 : i32, ssbuffer.nesting_depth = 3 : i32} : tensor<64x128xi1>, tensor<64x128xf32>
          %post_cnt_dep1 = arith.addi %counter1, %c1_i32_1 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.id=2 : i32, ssbuffer.nesting_depth = 3 : i32} : i32
          
          %41 = arith.mulf %arg10, %selected_dep1 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=2} : tensor<64x128xf32>
          hivm.hir.sync_block_wait {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=2}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %memspacecast_6 = memref.memory_space_cast %alloc_5 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=2} : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
          %42 = bufferization.to_tensor %memspacecast_6 restrict writable {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=2} : memref<64x128xf32>
          %43 = arith.addf %42, %41 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=2} : tensor<64x128xf32>
          %44 = tt.advance %arg12, [%c64_i32, %c0_i32] {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=2} : <tensor<64x128xf16>>
          %45 = tt.advance %arg13, [%c64_i32, %c0_i32] {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=2} : <tensor<64x128xf16>>
          hivm.hir.sync_block_set {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=2}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
          scf.yield {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=2} %38, %43, %25, %44, %45, %new_buf0_dep1, %new_buf1_dep1, %counter_after_dep1, %post_cnt_dep1  : tensor<64xf32>, tensor<64x128xf32>, tensor<64xf32>, !tt.ptr<tensor<64x128xf16>>, !tt.ptr<tensor<64x128xf16>>, tensor<64x128xf32>, tensor<64x128xf32>, i32, i32
        } {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32, ssbuffer.mainloop}
        %18 = tt.expand_dims %17#0 {axis = 1 : i32, ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %19 = tt.broadcast %18 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64x1xf32> -> tensor<64x128xf32>
        %20 = arith.divf %17#1, %19 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64x128xf32>
        %21 = arith.truncf %20 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64x128xf32> to tensor<64x128xf16>
        tt.store %16, %21 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : !tt.ptr<tensor<64x128xf16>>
      } {ssbuffer.end_nesting_depth = 1 : i32, ssbuffer.nesting_depth = 1 : i32}
      hivm.hir.sync_block_wait {ssbuffer.end_nesting_depth = 1 : i32, ssbuffer.nesting_depth = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 6
      scope.return {ssbuffer.end_nesting_depth = 1 : i32, ssbuffer.nesting_depth = 1 : i32}
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32}
    scope.scope : () -> () {
      hivm.hir.sync_block_set {ssbuffer.end_nesting_depth = 1 : i32, ssbuffer.nesting_depth = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 6
      scf.for %arg7 = %0 to %c131072_i32 step %c28_i32  : i32 {
        %1 = arith.divsi %arg7, %c128_i32 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i32
        %2 = arith.remsi %arg7, %c128_i32 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i32
        %3 = arith.divsi %1, %c8_i32 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i32
        %4 = arith.remsi %1, %c8_i32 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i32
        %5 = arith.extsi %3 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i32 to i64
        %6 = arith.muli %5, %c8388608_i64 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i64
        %7 = arith.extsi %4 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i32 to i64
        %8 = arith.muli %7, %c1048576_i64 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i64
        %9 = arith.addi %6, %8 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i64
        %10 = tt.addptr %arg0, %9 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : !tt.ptr<f16>, i64
        %11 = arith.muli %2, %c64_i32 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : i32
        %12 = tt.make_tensor_ptr %10, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%11, %c0_i32] {order = array<i32: 1, 0>, ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : <tensor<64x128xf16>>
        %13 = tt.addptr %arg2, %9 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : !tt.ptr<f16>, i64
        %14 = tt.make_tensor_ptr %13, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>, ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : <tensor<64x128xf16>>
        %15 = tt.addptr %arg1, %9 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : !tt.ptr<f16>, i64
        %16 = tt.make_tensor_ptr %15, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>, ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : <tensor<64x128xf16>>
        %17 = tt.load %12 {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : !tt.ptr<tensor<64x128xf16>>
        %18:2 = scf.for %arg8 = %c0_i32 to %c8192_i32 step %c64_i32 iter_args(%arg9 = %14, %arg10 = %16) -> (!tt.ptr<tensor<64x128xf16>>, !tt.ptr<tensor<64x128xf16>>)  : i32 {
          %19 = tt.load %arg10 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : !tt.ptr<tensor<64x128xf16>>
          %20 = tt.trans %19 {order = array<i32: 1, 0>, ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64x128xf16> -> tensor<128x64xf16>
          %21 = tt.dot %17, %20, %cst_1 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>
          hivm.hir.sync_block_wait {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} ins(%21 : tensor<64x64xf32>) outs(%alloc_4 : memref<64x64xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
          %22 = tt.load %arg9 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : !tt.ptr<tensor<64x128xf16>>
          hivm.hir.sync_block_wait {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          %23 = hivm.hir.convert_layout %alloc output_shape [64, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>, ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : (memref<4x4x16x16xf16, #hivm.address_space<cbuf>>) -> memref<64x64xf16, #hivm.address_space<cbuf>>
          %memspacecast = memref.memory_space_cast %23 {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : memref<64x64xf16, #hivm.address_space<cbuf>> to memref<64x64xf16>
          %24 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : memref<64x64xf16>
          %25 = tt.dot %24, %22, %cst {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
          hivm.hir.sync_block_set {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 6
          hivm.hir.sync_block_wait {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} ins(%25 : tensor<64x128xf32>) outs(%alloc_5 : memref<64x128xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %26 = tt.advance %arg9, [%c64_i32, %c0_i32] {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : <tensor<64x128xf16>>
          %27 = tt.advance %arg10, [%c64_i32, %c0_i32] {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} : <tensor<64x128xf16>>
          scf.yield {ssbuffer.end_nesting_depth = 3 : i32, ssbuffer.nesting_depth = 3 : i32, ssbuffer.id=1} %26, %27 : !tt.ptr<tensor<64x128xf16>>, !tt.ptr<tensor<64x128xf16>>
        } {ssbuffer.end_nesting_depth = 2 : i32, ssbuffer.nesting_depth = 2 : i32, ssbuffer.mainloop}
      } {ssbuffer.end_nesting_depth = 1 : i32, ssbuffer.nesting_depth = 1 : i32}
      hivm.hir.sync_block_wait {ssbuffer.end_nesting_depth = 1 : i32, ssbuffer.nesting_depth = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
      hivm.hir.sync_block_wait {ssbuffer.end_nesting_depth = 1 : i32, ssbuffer.nesting_depth = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
      scope.return {ssbuffer.end_nesting_depth = 1 : i32, ssbuffer.nesting_depth = 1 : i32}
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>, ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32}
    tt.return {ssbuffer.end_nesting_depth = 0 : i32, ssbuffer.nesting_depth = 0 : i32}
  }
}

