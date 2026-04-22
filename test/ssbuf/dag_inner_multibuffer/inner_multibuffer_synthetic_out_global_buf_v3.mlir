module {
  tt.func @inner_multibuffer_test(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant {ssbuffer.nesting_depth = 0 : i32} 0 : index
    %c64 = arith.constant {ssbuffer.nesting_depth = 0 : i32} 64 : index
    %c1024 = arith.constant {ssbuffer.nesting_depth = 0 : i32} 1024 : index
    %cst = arith.constant {ssbuffer.nesting_depth = 0 : i32} dense<0.000000e+00> : tensor<64xf32>
    scope.scope : () -> () {
      %alloc = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32>
      %0 = bufferization.to_tensor %memspacecast : memref<64xf32>
      %alloc_0 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      %memspacecast_1 = memref.memory_space_cast %alloc_0 : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32>
      %1 = bufferization.to_tensor %memspacecast_1 : memref<64xf32>
      %alloc_2 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      %memspacecast_3 = memref.memory_space_cast %alloc_2 : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32>
      %2 = bufferization.to_tensor %memspacecast_3 : memref<64xf32>
      %alloc_4 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      %memspacecast_5 = memref.memory_space_cast %alloc_4 : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32>
      %3 = bufferization.to_tensor %memspacecast_5 : memref<64xf32>
      %4 = scf.for %arg3 = %c0 to %c1024 step %c64 iter_args(%arg4 = %cst) -> (tensor<64xf32>) {
        %cst_6 = arith.constant {ssbuffer.core_type = "vector", ssbuffer.id = 0 : i32, ssbuffer.nesting_depth = 2 : i32} dense<1.000000e+00> : tensor<64xf32>
        %5 = arith.addf %cst_6, %cst_6 {ssbuffer.core_type = "vector", ssbuffer.id = 0 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64xf32>
        %6 = arith.subi %arg3, %c0 : index
        %7 = arith.divui %6, %c64 : index
        %8 = arith.index_cast %7 {ssbuffer.id = 0 : i32} : index to i32
        %c2_i32 = arith.constant {ssbuffer.id = 0 : i32} 2 : i32
        %9 = arith.remsi %8, %c2_i32 {ssbuffer.id = 0 : i32} : i32
        %c0_i32 = arith.constant {ssbuffer.id = 0 : i32} 0 : i32
        %10 = arith.cmpi eq, %9, %c0_i32 {ssbuffer.id = 0 : i32} : i32
        scf.if %10 {
          bufferization.materialize_in_destination %5 in writable %memspacecast_3 {ssbuffer.id = 0 : i32} : (tensor<64xf32>, memref<64xf32>) -> ()
        } {ssbuffer.id = 0 : i32}
        %c1_i32 = arith.constant {ssbuffer.id = 0 : i32} 1 : i32
        %11 = arith.cmpi eq, %9, %c1_i32 {ssbuffer.id = 0 : i32} : i32
        scf.if %11 {
          bufferization.materialize_in_destination %5 in writable %memspacecast_5 {ssbuffer.id = 0 : i32} : (tensor<64xf32>, memref<64xf32>) -> ()
        } {ssbuffer.id = 0 : i32}
        %12 = arith.mulf %arg4, %arg4 {ssbuffer.core_type = "vector", ssbuffer.id = 0 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64xf32>
        %13 = arith.subi %arg3, %c0 : index
        %14 = arith.divui %13, %c64 : index
        %15 = arith.index_cast %14 {ssbuffer.id = 0 : i32} : index to i32
        %c2_i32_7 = arith.constant {ssbuffer.id = 0 : i32} 2 : i32
        %16 = arith.remsi %15, %c2_i32_7 {ssbuffer.id = 0 : i32} : i32
        %c0_i32_8 = arith.constant {ssbuffer.id = 0 : i32} 0 : i32
        %17 = arith.cmpi eq, %16, %c0_i32_8 {ssbuffer.id = 0 : i32} : i32
        scf.if %17 {
          bufferization.materialize_in_destination %12 in writable %memspacecast {ssbuffer.id = 0 : i32} : (tensor<64xf32>, memref<64xf32>) -> ()
        } {ssbuffer.id = 0 : i32}
        %c1_i32_9 = arith.constant {ssbuffer.id = 0 : i32} 1 : i32
        %18 = arith.cmpi eq, %16, %c1_i32_9 {ssbuffer.id = 0 : i32} : i32
        scf.if %18 {
          bufferization.materialize_in_destination %12 in writable %memspacecast_1 {ssbuffer.id = 0 : i32} : (tensor<64xf32>, memref<64xf32>) -> ()
        } {ssbuffer.id = 0 : i32}
        %19 = arith.subi %arg3, %c0 : index
        %20 = arith.divui %19, %c64 : index
        %21 = arith.index_cast %20 {ssbuffer.id = 0 : i32} : index to i32
        %c2_i32_10 = arith.constant {ssbuffer.id = 0 : i32} 2 : i32
        %c1_i32_11 = arith.constant {ssbuffer.id = 0 : i32} 1 : i32
        %22 = arith.addi %21, %c1_i32_11 {ssbuffer.id = 0 : i32} : i32
        %23 = arith.remsi %22, %c2_i32_10 {ssbuffer.id = 0 : i32} : i32
        %c0_i32_12 = arith.constant {ssbuffer.id = 0 : i32} 0 : i32
        %24 = arith.cmpi eq, %23, %c0_i32_12 {ssbuffer.id = 0 : i32} : i32
        %25 = scf.if %24 -> (tensor<64xf32>) {
          %36 = bufferization.to_tensor %memspacecast_3 {ssbuffer.id = 0 : i32} : memref<64xf32>
          scf.yield %36 : tensor<64xf32>
        } else {
          %36 = bufferization.to_tensor %memspacecast_5 {ssbuffer.id = 0 : i32} : memref<64xf32>
          scf.yield %36 : tensor<64xf32>
        } {ssbuffer.id = 0 : i32}
        %26 = arith.addf %25, %25 {ssbuffer.core_type = "vector", ssbuffer.id = 1 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64xf32>
        %27 = arith.subi %arg3, %c0 : index
        %28 = arith.divui %27, %c64 : index
        %29 = arith.index_cast %28 {ssbuffer.id = 0 : i32} : index to i32
        %c2_i32_13 = arith.constant {ssbuffer.id = 0 : i32} 2 : i32
        %c1_i32_14 = arith.constant {ssbuffer.id = 0 : i32} 1 : i32
        %30 = arith.addi %29, %c1_i32_14 {ssbuffer.id = 0 : i32} : i32
        %31 = arith.remsi %30, %c2_i32_13 {ssbuffer.id = 0 : i32} : i32
        %c0_i32_15 = arith.constant {ssbuffer.id = 0 : i32} 0 : i32
        %32 = arith.cmpi eq, %31, %c0_i32_15 {ssbuffer.id = 0 : i32} : i32
        %33 = scf.if %32 -> (tensor<64xf32>) {
          %36 = bufferization.to_tensor %memspacecast {ssbuffer.id = 0 : i32} : memref<64xf32>
          scf.yield %36 : tensor<64xf32>
        } else {
          %36 = bufferization.to_tensor %memspacecast_1 {ssbuffer.id = 0 : i32} : memref<64xf32>
          scf.yield %36 : tensor<64xf32>
        } {ssbuffer.id = 0 : i32}
        %34 = arith.addf %33, %33 {ssbuffer.core_type = "vector", ssbuffer.id = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64xf32>
        %35 = arith.addf %34, %34 {ssbuffer.core_type = "vector", ssbuffer.id = 2 : i32, ssbuffer.nesting_depth = 2 : i32} : tensor<64xf32>
        scf.yield %35 : tensor<64xf32>
      } {ssbuffer.mainloop, ssbuffer.nesting_depth = 2 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, ssbuffer.nesting_depth = 1 : i32}
    tt.return
  }
}

