module {
  tt.func @inner_multibuffer_test(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<tensor<64x64xf32>>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant {ssbuffer.nesting_depth = 0 : i32} 0 : index
    %c64 = arith.constant {ssbuffer.nesting_depth = 0 : i32} 64 : index
    %c1024 = arith.constant {ssbuffer.nesting_depth = 0 : i32} 1024 : index
    %cst = arith.constant {ssbuffer.nesting_depth = 0 : i32} dense<0.000000e+00> : tensor<64x64xf32>
    scope.scope : () -> () {
      %alloc = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
      %0 = bufferization.to_tensor %memspacecast : memref<64x64xf32>
      %alloc_0 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
      %memspacecast_1 = memref.memory_space_cast %alloc_0 : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
      %1 = bufferization.to_tensor %memspacecast_1 : memref<64x64xf32>
      %alloc_2 = memref.alloc() : memref<64x64xf32, #hivm.address_space<cbuf>>
      %memspacecast_3 = memref.memory_space_cast %alloc_2 : memref<64x64xf32, #hivm.address_space<cbuf>> to memref<64x64xf32>
      %2 = bufferization.to_tensor %memspacecast_3 : memref<64x64xf32>
      %alloc_4 = memref.alloc() : memref<64x64xf32, #hivm.address_space<cbuf>>
      %memspacecast_5 = memref.memory_space_cast %alloc_4 : memref<64x64xf32, #hivm.address_space<cbuf>> to memref<64x64xf32>
      %3 = bufferization.to_tensor %memspacecast_5 : memref<64x64xf32>
      %4 = scf.for %arg3 = %c0 to %c1024 step %c64 iter_args(%arg4 = %cst) -> (tensor<64x64xf32>) {
        %cst_6 = arith.constant {ssbuffer.core_type = "vector", ssbuffer.id = 0 : i32, ssbuffer.nesting_depth = 2 : i32} dense<1.000000e+00> : tensor<64x64xf32>
        %5 = arith.subi %arg3, %c0 : index
        %6 = arith.divui %5, %c64 : index
        %7 = arith.index_cast %6 {ssbuffer.id = 0 : i32} : index to i32
        %c2_i32 = arith.constant {ssbuffer.id = 0 : i32} 2 : i32
        %8 = arith.remsi %7, %c2_i32 {ssbuffer.id = 0 : i32} : i32
        %c0_i32 = arith.constant {ssbuffer.id = 0 : i32} 0 : i32
        %9 = arith.cmpi eq, %8, %c0_i32 {ssbuffer.id = 0 : i32} : i32
        scf.if %9 {
          bufferization.materialize_in_destination %cst_6 in writable %memspacecast {ssbuffer.id = 0 : i32} : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
        } {ssbuffer.id = 0 : i32}
        %c1_i32 = arith.constant {ssbuffer.id = 0 : i32} 1 : i32
        %10 = arith.cmpi eq, %8, %c1_i32 {ssbuffer.id = 0 : i32} : i32
        scf.if %10 {
          bufferization.materialize_in_destination %cst_6 in writable %memspacecast_1 {ssbuffer.id = 0 : i32} : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
        } {ssbuffer.id = 0 : i32}
        %cst_7 = arith.constant {ssbuffer.core_type = "cube", ssbuffer.id = 1 : i32, ssbuffer.nesting_depth = 2 : i32} dense<0.000000e+00> : tensor<64x64xf32>
        %11 = tt.load %arg1 {ssbuffer.core_type = "cube", ssbuffer.id = 1 : i32, ssbuffer.nesting_depth = 2 : i32} : !tt.ptr<tensor<64x64xf32>>
        %12 = arith.subi %arg3, %c0 : index
        %13 = arith.divui %12, %c64 : index
        %14 = arith.index_cast %13 {ssbuffer.id = 0 : i32} : index to i32
        %c2_i32_8 = arith.constant {ssbuffer.id = 0 : i32} 2 : i32
        %c1_i32_9 = arith.constant {ssbuffer.id = 0 : i32} 1 : i32
        %15 = arith.addi %14, %c1_i32_9 {ssbuffer.id = 0 : i32} : i32
        %16 = arith.remsi %15, %c2_i32_8 {ssbuffer.id = 0 : i32} : i32
        %c0_i32_10 = arith.constant {ssbuffer.id = 0 : i32} 0 : i32
        %17 = arith.cmpi eq, %16, %c0_i32_10 {ssbuffer.id = 0 : i32} : i32
        %18 = scf.if %17 -> (tensor<64x64xf32>) {
          %43 = bufferization.to_tensor %memspacecast {ssbuffer.id = 0 : i32} : memref<64x64xf32>
          scf.yield %43 : tensor<64x64xf32>
        } else {
          %43 = bufferization.to_tensor %memspacecast_1 {ssbuffer.id = 0 : i32} : memref<64x64xf32>
          scf.yield %43 : tensor<64x64xf32>
        } {ssbuffer.id = 0 : i32}
        %19 = tt.dot %18, %11, %cst_7 {ssbuffer.core_type = "cube", ssbuffer.id = 1 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>
        %20 = arith.subi %arg3, %c0 : index
        %21 = arith.divui %20, %c64 : index
        %22 = arith.index_cast %21 {ssbuffer.id = 1 : i32} : index to i32
        %c2_i32_11 = arith.constant {ssbuffer.id = 1 : i32} 2 : i32
        %23 = arith.remsi %22, %c2_i32_11 {ssbuffer.id = 1 : i32} : i32
        %c0_i32_12 = arith.constant {ssbuffer.id = 1 : i32} 0 : i32
        %24 = arith.cmpi eq, %23, %c0_i32_12 {ssbuffer.id = 1 : i32} : i32
        scf.if %24 {
          bufferization.materialize_in_destination %19 in writable %memspacecast_3 {ssbuffer.id = 1 : i32} : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
        } {ssbuffer.id = 1 : i32}
        %c1_i32_13 = arith.constant {ssbuffer.id = 1 : i32} 1 : i32
        %25 = arith.cmpi eq, %23, %c1_i32_13 {ssbuffer.id = 1 : i32} : i32
        scf.if %25 {
          bufferization.materialize_in_destination %19 in writable %memspacecast_5 {ssbuffer.id = 1 : i32} : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
        } {ssbuffer.id = 1 : i32}
        %26 = arith.subi %arg3, %c0 : index
        %27 = arith.divui %26, %c64 : index
        %28 = arith.index_cast %27 {ssbuffer.id = 0 : i32} : index to i32
        %c2_i32_14 = arith.constant {ssbuffer.id = 0 : i32} 2 : i32
        %c1_i32_15 = arith.constant {ssbuffer.id = 0 : i32} 1 : i32
        %29 = arith.addi %28, %c1_i32_15 {ssbuffer.id = 0 : i32} : i32
        %30 = arith.remsi %29, %c2_i32_14 {ssbuffer.id = 0 : i32} : i32
        %c0_i32_16 = arith.constant {ssbuffer.id = 0 : i32} 0 : i32
        %31 = arith.cmpi eq, %30, %c0_i32_16 {ssbuffer.id = 0 : i32} : i32
        %32 = scf.if %31 -> (tensor<64x64xf32>) {
          %43 = bufferization.to_tensor %memspacecast {ssbuffer.id = 0 : i32} : memref<64x64xf32>
          scf.yield %43 : tensor<64x64xf32>
        } else {
          %43 = bufferization.to_tensor %memspacecast_1 {ssbuffer.id = 0 : i32} : memref<64x64xf32>
          scf.yield %43 : tensor<64x64xf32>
        } {ssbuffer.id = 0 : i32}
        %33 = arith.addf %32, %32 {ssbuffer.core_type = "vector", ssbuffer.id = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64x64xf32>
        %34 = arith.subi %arg3, %c0 : index
        %35 = arith.divui %34, %c64 : index
        %36 = arith.index_cast %35 {ssbuffer.id = 1 : i32} : index to i32
        %c2_i32_17 = arith.constant {ssbuffer.id = 1 : i32} 2 : i32
        %c1_i32_18 = arith.constant {ssbuffer.id = 1 : i32} 1 : i32
        %37 = arith.addi %36, %c1_i32_18 {ssbuffer.id = 1 : i32} : i32
        %38 = arith.remsi %37, %c2_i32_17 {ssbuffer.id = 1 : i32} : i32
        %c0_i32_19 = arith.constant {ssbuffer.id = 1 : i32} 0 : i32
        %39 = arith.cmpi eq, %38, %c0_i32_19 {ssbuffer.id = 1 : i32} : i32
        %40 = scf.if %39 -> (tensor<64x64xf32>) {
          %43 = bufferization.to_tensor %memspacecast_3 {ssbuffer.id = 1 : i32} : memref<64x64xf32>
          scf.yield %43 : tensor<64x64xf32>
        } else {
          %43 = bufferization.to_tensor %memspacecast_5 {ssbuffer.id = 1 : i32} : memref<64x64xf32>
          scf.yield %43 : tensor<64x64xf32>
        } {ssbuffer.id = 1 : i32}
        %41 = arith.addf %40, %40 {ssbuffer.core_type = "vector", ssbuffer.id = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64x64xf32>
        %42 = arith.addf %33, %41 {ssbuffer.core_type = "vector", ssbuffer.id = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64x64xf32>
        scf.yield %42 : tensor<64x64xf32>
      } {ssbuffer.mainloop, ssbuffer.nesting_depth = 2 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, ssbuffer.nesting_depth = 1 : i32}
    tt.return
  }
}

