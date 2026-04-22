module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_parallel_hstu_attn_bwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg12: f32, %arg13: f32, %arg14: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x2x16x16xf16>
    %c1_i32 = arith.constant 1 : i32
    %c2147483647_i64 = arith.constant 2147483647 : i64
    %c-2147483648_i64 = arith.constant -2147483648 : i64
    %c4_i64 = arith.constant 4 : i64
    %c64_i64 = arith.constant 64 : i64
    %c256_i64 = arith.constant 256 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %cst_0 = arith.constant dense<128> : tensor<32x1xi64>
    %cst_1 = arith.constant dense<256> : tensor<32x1xi64>
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %true = arith.constant true
    %c31_i64 = arith.constant 31 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x32xf16>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x64xf16>
    %c255_i64 = arith.constant 255 : i64
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<256x32xf16>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<256x64xf16>
    %cst_6 = arith.constant dense<128> : tensor<256x1xi64>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<32x256xf32>
    %cst_8 = arith.constant dense<256> : tensor<256x1xi64>
    %cst_9 = arith.constant dense<1.000000e+00> : tensor<32x256xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<256x32xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<256x64xf32>
    %alloc = memref.alloc() : memref<16x2x16x16xf16, #hivm.address_space<cbuf>>
    %alloc_13 = memref.alloc() : memref<256x64xf32, #hivm.address_space<ub>>
    %alloc_14 = memref.alloc() : memref<16x2x16x16xf16, #hivm.address_space<cbuf>>
    %alloc_15 = memref.alloc() : memref<16x2x16x16xf16, #hivm.address_space<cbuf>>
    %alloc_16 = memref.alloc() : memref<32x256xf32, #hivm.address_space<ub>>
    %alloc_17 = memref.alloc() : memref<32x256xf32, #hivm.address_space<ub>>
    %alloc_18 = memref.alloc() : memref<256x32xf32, #hivm.address_space<ub>>
    %0 = tt.get_program_id x : i32
    %c0_i32_19 = arith.constant 0 : i32
    %c0_i32_20 = arith.constant 0 : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(1024 : i64) : i64
    %4 = llvm.inttoptr %2 : i64 to !llvm.ptr<11>
    %5 = llvm.inttoptr %3 : i64 to !llvm.ptr<11>
    llvm.store %1, %4 : i32, !llvm.ptr<11>
    llvm.store %1, %5 : i32, !llvm.ptr<11>
    %6 = llvm.mlir.constant(4 : i64) : i64
    %7 = llvm.mlir.constant(1028 : i64) : i64
    %8 = llvm.inttoptr %6 : i64 to !llvm.ptr<11>
    %9 = llvm.inttoptr %7 : i64 to !llvm.ptr<11>
    llvm.store %1, %8 : i32, !llvm.ptr<11>
    llvm.store %1, %9 : i32, !llvm.ptr<11>
    %10 = llvm.mlir.constant(8 : i64) : i64
    %11 = llvm.mlir.constant(1032 : i64) : i64
    %12 = llvm.inttoptr %10 : i64 to !llvm.ptr<11>
    %13 = llvm.inttoptr %11 : i64 to !llvm.ptr<11>
    llvm.store %1, %12 : i32, !llvm.ptr<11>
    llvm.store %1, %13 : i32, !llvm.ptr<11>
    %14 = llvm.mlir.constant(12 : i64) : i64
    %15 = llvm.mlir.constant(1036 : i64) : i64
    %16 = llvm.inttoptr %14 : i64 to !llvm.ptr<11>
    %17 = llvm.inttoptr %15 : i64 to !llvm.ptr<11>
    llvm.store %1, %16 : i32, !llvm.ptr<11>
    llvm.store %1, %17 : i32, !llvm.ptr<11>
    %18 = llvm.mlir.constant(16 : i64) : i64
    %19 = llvm.mlir.constant(1040 : i64) : i64
    %20 = llvm.inttoptr %18 : i64 to !llvm.ptr<11>
    %21 = llvm.inttoptr %19 : i64 to !llvm.ptr<11>
    llvm.store %1, %20 : i32, !llvm.ptr<11>
    llvm.store %1, %21 : i32, !llvm.ptr<11>
    %22 = llvm.mlir.constant(20 : i64) : i64
    %23 = llvm.mlir.constant(1044 : i64) : i64
    %24 = llvm.inttoptr %22 : i64 to !llvm.ptr<11>
    %25 = llvm.inttoptr %23 : i64 to !llvm.ptr<11>
    llvm.store %1, %24 : i32, !llvm.ptr<11>
    llvm.store %1, %25 : i32, !llvm.ptr<11>
    %26 = llvm.mlir.constant(24 : i64) : i64
    %27 = llvm.mlir.constant(1048 : i64) : i64
    %28 = llvm.inttoptr %26 : i64 to !llvm.ptr<11>
    %29 = llvm.inttoptr %27 : i64 to !llvm.ptr<11>
    llvm.store %1, %28 : i32, !llvm.ptr<11>
    llvm.store %1, %29 : i32, !llvm.ptr<11>
    %30 = llvm.mlir.constant(0 : i64) : i64
    %31 = llvm.mlir.constant(32 : i64) : i64
    %32 = llvm.mlir.constant(64 : i64) : i64
    %33 = llvm.mlir.constant(96 : i64) : i64
    %34 = llvm.mlir.constant(0 : i32) : i32
    %35 = llvm.inttoptr %30 : i64 to !llvm.ptr<11>
    %36 = llvm.inttoptr %31 : i64 to !llvm.ptr<11>
    %37 = llvm.inttoptr %32 : i64 to !llvm.ptr<11>
    %38 = llvm.inttoptr %33 : i64 to !llvm.ptr<11>
    llvm.store %34, %35 : i32, !llvm.ptr<11>
    llvm.store %34, %36 : i32, !llvm.ptr<11>
    llvm.store %34, %37 : i32, !llvm.ptr<11>
    llvm.store %34, %38 : i32, !llvm.ptr<11>
    %c0_i64_21 = arith.constant 0 : i64
    scope.scope : () -> () {
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 14
      %39 = llvm.mlir.constant(32 : i64) : i64
      %40 = llvm.mlir.constant(0 : i64) : i64
      %41 = llvm.mlir.constant(0 : i32) : i32
      %42 = llvm.mlir.constant(1 : i32) : i32
      %43 = hivm.hir.get_sub_block_idx -> i64
      %44 = arith.muli %43, %39 : i64
      %45 = arith.addi %44, %39 : i64
      %46 = arith.cmpi eq, %43, %40 : i64
      %c1_i32_22 = arith.constant 1 : i32
      %c1_i32_23 = arith.constant 1 : i32
      %c1_i32_24 = arith.constant 1 : i32
      %c1_i32_25 = arith.constant 1 : i32
      %c1_i32_26 = arith.constant 1 : i32
      %c1_i32_27 = arith.constant 1 : i32
      %c1_i32_28 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c2_i32_29 = arith.constant 2 : i32
      %c0_i32_30 = arith.constant 0 : i32
      %c1_i32_31 = arith.constant 1 : i32
      %c1024_i64 = arith.constant 1024 : i64
      %47 = hivm.hir.get_sub_block_idx -> i64
      %48 = arith.muli %47, %c1024_i64 : i64
      %c0_i64_32 = arith.constant 0 : i64
      %49 = arith.addi %c0_i64_32, %48 : i64
      %50 = llvm.inttoptr %49 : i64 to !llvm.ptr<11>
      %c4_i64_33 = arith.constant 4 : i64
      %51 = arith.addi %c4_i64_33, %48 : i64
      %52 = llvm.inttoptr %51 : i64 to !llvm.ptr<11>
      %c8_i64 = arith.constant 8 : i64
      %53 = arith.addi %c8_i64, %48 : i64
      %54 = llvm.inttoptr %53 : i64 to !llvm.ptr<11>
      %c12_i64 = arith.constant 12 : i64
      %55 = arith.addi %c12_i64, %48 : i64
      %56 = llvm.inttoptr %55 : i64 to !llvm.ptr<11>
      %c16_i64 = arith.constant 16 : i64
      %57 = arith.addi %c16_i64, %48 : i64
      %58 = llvm.inttoptr %57 : i64 to !llvm.ptr<11>
      %c20_i64 = arith.constant 20 : i64
      %59 = arith.addi %c20_i64, %48 : i64
      %60 = llvm.inttoptr %59 : i64 to !llvm.ptr<11>
      %c24_i64 = arith.constant 24 : i64
      %61 = arith.addi %c24_i64, %48 : i64
      %62 = llvm.inttoptr %61 : i64 to !llvm.ptr<11>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
      %63 = tt.get_num_programs x : i32
      %64 = scf.for %arg15 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg16 = %c0_i64) -> (i64)  : i32 {
        %105 = tt.addptr %arg10, %arg15 : !tt.ptr<i64>, i32
        %106 = tt.addptr %105, %c1_i32 : !tt.ptr<i64>, i32
        %107 = tt.load %106 : !tt.ptr<i64>
        %108 = tt.load %105 : !tt.ptr<i64>
        %109 = arith.subi %107, %108 : i64
        tt.assert %true, "int32 overflow detected for operation sub" : i1
        %110 = arith.addi %109, %c31_i64 : i64
        %111 = arith.divsi %110, %c32_i64 : i64
        %112 = arith.addi %arg16, %111 : i64
        scf.yield %112 : i64
      }
      %65 = arith.muli %64, %c4_i64 : i64
      %66 = arith.extsi %63 : i32 to i64
      %67 = arith.subi %66, %c1_i64 : i64
      %68 = arith.cmpi sle, %67, %c2147483647_i64 : i64
      %69 = arith.cmpi sge, %67, %c-2147483648_i64 : i64
      %70 = arith.andi %68, %69 : i1
      tt.assert %70, "int32 overflow detected for operation sub" : i1
      %71 = arith.subi %63, %c1_i32 : i32
      %72 = arith.extsi %71 : i32 to i64
      %73 = arith.addi %65, %72 : i64
      %74 = arith.divsi %73, %66 : i64
      %75 = arith.extsi %0 : i32 to i64
      %76 = arith.muli %75, %74 : i64
      %77 = arith.addi %0, %c1_i32 : i32
      %78 = arith.extsi %77 : i32 to i64
      %79 = arith.muli %78, %74 : i64
      %80 = arith.minsi %79, %65 : i64
      %81 = arith.remsi %76, %64 : i64
      %82:3 = scf.for %arg15 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64) -> (i64, i64, i64)  : i32 {
        %105 = tt.addptr %arg10, %arg15 : !tt.ptr<i64>, i32
        %106 = tt.addptr %105, %c1_i32 : !tt.ptr<i64>, i32
        %107 = tt.load %106 : !tt.ptr<i64>
        %108 = tt.load %105 : !tt.ptr<i64>
        %109 = arith.subi %107, %108 : i64
        tt.assert %true, "int32 overflow detected for operation sub" : i1
        %110 = arith.addi %109, %c31_i64 : i64
        %111 = arith.divsi %110, %c32_i64 : i64
        %112 = arith.cmpi sge, %81, %arg18 : i64
        %113 = arith.extsi %arg15 : i32 to i64
        %114 = arith.select %112, %113, %arg16 : i64
        %115 = arith.select %112, %arg18, %arg17 : i64
        %116 = arith.addi %arg18, %111 : i64
        scf.yield %114, %115, %116 : i64, i64, i64
      }
      %83 = tt.addptr %arg10, %82#0 : !tt.ptr<i64>, i64
      %84 = tt.addptr %83, %c1_i32 : !tt.ptr<i64>, i32
      %85 = tt.load %84 : !tt.ptr<i64>
      %86 = tt.load %83 : !tt.ptr<i64>
      %87 = arith.subi %85, %86 : i64
      tt.assert %true, "int32 overflow detected for operation sub" : i1
      %88 = arith.addi %87, %c31_i64 : i64
      %89 = arith.divsi %88, %c32_i64 : i64
      %90 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %91 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
      %92 = arith.extsi %90 : tensor<32xi32> to tensor<32xi64>
      %93 = tt.expand_dims %90 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
      %94 = tt.expand_dims %91 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %95 = tt.broadcast %94 : tensor<1x64xi32> -> tensor<32x64xi32>
      %96 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %97 = arith.extsi %96 : tensor<256xi32> to tensor<256xi64>
      %98 = tt.splat %arg12 : f32 -> tensor<32x256xf32>
      %99 = tt.splat %arg13 : f32 -> tensor<32x256xf32>
      %100 = arith.extsi %94 : tensor<1x64xi32> to tensor<1x64xi64>
      %101 = tt.broadcast %100 : tensor<1x64xi64> -> tensor<256x64xi64>
      %102 = arith.extsi %93 : tensor<1x32xi32> to tensor<1x32xi64>
      %103 = tt.broadcast %102 : tensor<1x32xi64> -> tensor<256x32xi64>
      %104:3 = scf.for %arg15 = %76 to %80 step %c1_i64 iter_args(%arg16 = %82#0, %arg17 = %82#1, %arg18 = %89) -> (i64, i64, i64)  : i64 {
        %105 = arith.divsi %arg15, %64 : i64
        %106 = arith.remsi %arg15, %64 : i64
        %107 = arith.cmpi slt, %106, %arg17 : i64
        %108 = arith.select %107, %c0_i64, %arg16 : i64
        %109 = arith.select %107, %c0_i64, %arg17 : i64
        %110 = scf.if %107 -> (i64) {
          %164 = tt.addptr %arg10, %c1_i32 : !tt.ptr<i64>, i32
          %165 = tt.load %164 : !tt.ptr<i64>
          %166 = tt.load %arg10 : !tt.ptr<i64>
          %167 = arith.subi %165, %166 : i64
          tt.assert %true, "int32 overflow detected for operation sub" : i1
          %168 = arith.addi %167, %c31_i64 : i64
          %169 = arith.divsi %168, %c32_i64 : i64
          scf.yield %169 : i64
        } else {
          scf.yield %arg18 : i64
        }
        %111:3 = scf.while (%arg19 = %109, %arg20 = %108, %arg21 = %110) : (i64, i64, i64) -> (i64, i64, i64) {
          %164 = arith.addi %arg19, %arg21 : i64
          %165 = arith.cmpi sge, %106, %164 : i64
          scf.condition(%165) %arg19, %arg20, %arg21 : i64, i64, i64
        } do {
        ^bb0(%arg19: i64, %arg20: i64, %arg21: i64):
          %164 = arith.addi %arg19, %arg21 : i64
          %165 = arith.addi %arg20, %c1_i64 : i64
          %166 = tt.addptr %arg10, %165 : !tt.ptr<i64>, i64
          %167 = tt.addptr %166, %c1_i32 : !tt.ptr<i64>, i32
          %168 = tt.load %167 : !tt.ptr<i64>
          %169 = tt.load %166 : !tt.ptr<i64>
          %170 = arith.subi %168, %169 : i64
          tt.assert %true, "int32 overflow detected for operation sub" : i1
          %171 = arith.addi %170, %c31_i64 : i64
          %172 = arith.divsi %171, %c32_i64 : i64
          scf.yield %164, %165, %172 : i64, i64, i64
        }
        %112 = arith.subi %106, %111#0 : i64
        %113 = tt.addptr %arg10, %111#1 : !tt.ptr<i64>, i64
        %114 = tt.load %113 : !tt.ptr<i64>
        %115 = tt.addptr %113, %c1_i32 : !tt.ptr<i64>, i32
        %116 = tt.load %115 : !tt.ptr<i64>
        %117 = tt.addptr %arg11, %111#1 : !tt.ptr<i64>, i64
        %118 = tt.load %117 : !tt.ptr<i64>
        %119 = tt.addptr %117, %c1_i32 : !tt.ptr<i64>, i32
        %120 = tt.load %119 : !tt.ptr<i64>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
        %121 = arith.subi %116, %114 : i64
        %122 = arith.subi %120, %118 : i64
        %123 = arith.muli %105, %c64_i64 : i64
        %124 = arith.muli %114, %c256_i64 : i64
        %125 = arith.addi %123, %124 : i64
        %126 = arith.muli %118, %c256_i64 : i64
        %127 = arith.addi %123, %126 : i64
        %128 = arith.muli %105, %c32_i64 : i64
        %129 = arith.muli %118, %c128_i64 : i64
        %130 = arith.addi %128, %129 : i64
        %131 = tt.addptr %arg4, %125 : !tt.ptr<f32>, i64
        %132 = tt.addptr %arg7, %125 : !tt.ptr<f16>, i64
        %133 = tt.addptr %arg8, %127 : !tt.ptr<f16>, i64
        %134 = tt.addptr %arg9, %130 : !tt.ptr<f16>, i64
        %135 = arith.muli %112, %c32_i64 : i64
        %136 = tt.splat %135 : i64 -> tensor<32xi64>
        %137 = arith.addi %136, %92 : tensor<32xi64>
        %138 = tt.splat %121 : i64 -> tensor<32xi64>
        %139 = arith.cmpi slt, %137, %138 : tensor<32xi64>
        %140 = tt.expand_dims %137 {axis = 1 : i32} : tensor<32xi64> -> tensor<32x1xi64>
        %141 = tt.expand_dims %139 {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
        %142 = arith.muli %140, %cst_1 : tensor<32x1xi64>
        %143 = tt.broadcast %141 : tensor<32x1xi1> -> tensor<32x64xi1>
        tt.assert %true, "int32 overflow detected for operation sub" : i1
        %144 = arith.addi %122, %c255_i64 : i64
        %145 = arith.divsi %144, %c256_i64 : i64
        %146 = tt.splat %122 : i64 -> tensor<256xi64>
        %147 = tt.splat %133 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>>
        %148 = tt.splat %134 : !tt.ptr<f16> -> tensor<256x32x!tt.ptr<f16>>
        %c4_i64_34 = arith.constant 4 : i64
        %149 = arith.muli %c1_i64, %c4_i64_34 : i64
        %150 = arith.addi %145, %149 : i64
        %c4_i64_35 = arith.constant 4 : i64
        %151 = arith.muli %c1_i64, %c4_i64_35 : i64
        %152 = arith.subi %150, %151 : i64
        %153:13 = scf.for %arg19 = %c0_i64 to %150 step %c1_i64 iter_args(%arg20 = %cst, %arg21 = %c0_i32_19, %arg22 = %c0_i32_20, %arg23 = %c0_i64, %arg24 = %c0_i64, %arg25 = %c0_i64, %arg26 = %c0_i64, %arg27 = %cst, %arg28 = %c0_i64_21, %arg29 = %c0_i64_21, %arg30 = %cst, %arg31 = %c0_i64_21, %arg32 = %c0_i64_21) -> (tensor<16x2x16x16xf16>, i32, i32, i64, i64, i64, i64, tensor<16x2x16x16xf16>, i64, i64, tensor<16x2x16x16xf16>, i64, i64)  : i64 {
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
          %164 = llvm.load %50 : !llvm.ptr<11> -> i32
          %165 = llvm.load %52 : !llvm.ptr<11> -> i32
          %166 = llvm.load %54 : !llvm.ptr<11> -> i32
          %167 = llvm.load %56 : !llvm.ptr<11> -> i32
          %168 = llvm.load %58 : !llvm.ptr<11> -> i32
          %169 = llvm.load %60 : !llvm.ptr<11> -> i32
          %170 = llvm.load %62 : !llvm.ptr<11> -> i32
          %true_36 = arith.constant true
          %171 = arith.cmpi sgt, %168, %c0_i32_30 : i32
          %172 = arith.cmpi sgt, %169, %c0_i32_30 : i32
          %173 = arith.andi %172, %171 : i1
          %174 = arith.cmpi slt, %arg21, %c2_i32 : i32
          %175 = arith.andi %174, %173 : i1
          %176 = arith.cmpi slt, %arg22, %c2_i32_29 : i32
          %177 = arith.andi %176, %175 : i1
          %178 = arith.cmpi slt, %164, %c1_i32_22 : i32
          %179 = arith.andi %178, %177 : i1
          %180 = arith.cmpi slt, %arg23, %152 : i64
          %181 = arith.andi %179, %180 : i1
          %c0_i64_37 = arith.constant 0 : i64
          %c1_i64_38 = arith.constant 1 : i64
          %c2_i64 = arith.constant 2 : i64
          %c0_i64_39 = arith.constant 0 : i64
          %c1_i64_40 = arith.constant 1 : i64
          %c2_i64_41 = arith.constant 2 : i64
          %182:8 = scf.if %181 -> (tensor<16x2x16x16xf16>, i32, i32, i64, tensor<16x2x16x16xf16>, i64, tensor<16x2x16x16xf16>, i64) {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
            %201 = arith.muli %arg23, %c256_i64 : i64
            %202 = tt.splat %201 : i64 -> tensor<256xi64>
            %203 = arith.addi %202, %97 : tensor<256xi64>
            %204 = arith.cmpi slt, %203, %146 : tensor<256xi64>
            %205 = tt.expand_dims %203 {axis = 1 : i32} : tensor<256xi64> -> tensor<256x1xi64>
            %206 = arith.muli %205, %cst_8 : tensor<256x1xi64>
            %207 = tt.expand_dims %204 {axis = 1 : i32} : tensor<256xi1> -> tensor<256x1xi1>
            %208 = tt.broadcast %207 : tensor<256x1xi1> -> tensor<256x64xi1>
            %209 = arith.muli %205, %cst_6 : tensor<256x1xi64>
            %210 = tt.broadcast %207 : tensor<256x1xi1> -> tensor<256x32xi1>
            %memspacecast = memref.memory_space_cast %alloc_17 : memref<32x256xf32, #hivm.address_space<ub>> to memref<32x256xf32>
            %211 = bufferization.to_tensor %memspacecast restrict writable : memref<32x256xf32>
            %212 = arith.mulf %211, %98 : tensor<32x256xf32>
            %213 = arith.subf %cst_7, %212 : tensor<32x256xf32>
            %214 = math.exp %213 : tensor<32x256xf32>
            %215 = arith.addf %214, %cst_9 : tensor<32x256xf32>
            %216 = arith.divf %cst_9, %215 : tensor<32x256xf32>
            %217 = arith.mulf %212, %216 : tensor<32x256xf32>
            %218 = arith.mulf %217, %99 : tensor<32x256xf32>
            %219 = arith.truncf %218 : tensor<32x256xf32> to tensor<32x256xf16>
            %220 = tt.reshape %219 : tensor<32x256xf16> -> tensor<32x16x16xf16>
            annotation.mark %220 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<32x16x16xf16>
            %221 = tt.trans %220 {order = array<i32: 1, 0, 2>} : tensor<32x16x16xf16> -> tensor<16x32x16xf16>
            %222 = tt.reshape %221 : tensor<16x32x16xf16> -> tensor<16x2x16x16xf16>
            annotation.mark %222 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<16x2x16x16xf16>
            %memspacecast_45 = memref.memory_space_cast %alloc_16 : memref<32x256xf32, #hivm.address_space<ub>> to memref<32x256xf32>
            %223 = bufferization.to_tensor %memspacecast_45 restrict writable : memref<32x256xf32>
            %224 = arith.mulf %223, %99 : tensor<32x256xf32>
            %225 = arith.subf %cst_9, %216 : tensor<32x256xf32>
            %226 = arith.mulf %212, %225 : tensor<32x256xf32>
            %227 = arith.addf %226, %cst_9 : tensor<32x256xf32>
            %228 = arith.mulf %216, %227 : tensor<32x256xf32>
            %229 = arith.mulf %224, %228 : tensor<32x256xf32>
            %230 = arith.mulf %229, %98 : tensor<32x256xf32>
            %231 = arith.truncf %230 : tensor<32x256xf32> to tensor<32x256xf16>
            %232 = tt.reshape %231 : tensor<32x256xf16> -> tensor<32x16x16xf16>
            annotation.mark %232 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<32x16x16xf16>
            %233 = tt.trans %232 {order = array<i32: 1, 0, 2>} : tensor<32x16x16xf16> -> tensor<16x32x16xf16>
            %234 = tt.reshape %233 : tensor<16x32x16xf16> -> tensor<16x2x16x16xf16>
            %235 = arith.remsi %arg28, %c2_i64 : i64
            %236 = arith.cmpi eq, %235, %c0_i64_37 : i64
            %splat = tensor.splat %236 : tensor<16x2x16x16xi1>
            %237 = arith.select %splat, %234, %arg20 : tensor<16x2x16x16xi1>, tensor<16x2x16x16xf16>
            %238 = arith.remsi %arg31, %c2_i64_41 : i64
            %239 = arith.cmpi eq, %238, %c0_i64_39 : i64
            %splat_46 = tensor.splat %239 : tensor<16x2x16x16xi1>
            %240 = arith.select %splat_46, %237, %arg20 : tensor<16x2x16x16xi1>, tensor<16x2x16x16xf16>
            %241 = arith.select %splat_46, %arg30, %237 : tensor<16x2x16x16xi1>, tensor<16x2x16x16xf16>
            %242 = arith.addi %arg31, %c1_i64_40 : i64
            %243 = arith.select %splat, %arg27, %234 : tensor<16x2x16x16xi1>, tensor<16x2x16x16xf16>
            %244 = arith.addi %arg28, %c1_i64_38 : i64
            annotation.mark %234 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<16x2x16x16xf16>
            annotation.mark %232 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<32x16x16xf16>
            annotation.mark %234 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<16x2x16x16xf16>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 8
            hivm.hir.copy ins(%222 : tensor<16x2x16x16xf16>) outs(%alloc : memref<16x2x16x16xf16, #hivm.address_space<cbuf>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 8
            %245 = llvm.load %58 : !llvm.ptr<11> -> i32
            %246 = arith.subi %245, %c1_i32_31 : i32
            llvm.store %246, %58 : i32, !llvm.ptr<11>
            %247 = llvm.load %60 : !llvm.ptr<11> -> i32
            %248 = arith.subi %247, %c1_i32_31 : i32
            llvm.store %248, %60 : i32, !llvm.ptr<11>
            %249 = arith.addi %arg21, %c1_i32_31 : i32
            %250 = arith.addi %arg22, %c1_i32_31 : i32
            %251 = llvm.load %50 : !llvm.ptr<11> -> i32
            %252 = arith.addi %251, %c1_i32_31 : i32
            llvm.store %252, %50 : i32, !llvm.ptr<11>
            %253 = arith.addi %arg23, %c1_i64 : i64
            scf.yield %240, %249, %250, %253, %243, %244, %241, %242 : tensor<16x2x16x16xf16>, i32, i32, i64, tensor<16x2x16x16xf16>, i64, tensor<16x2x16x16xf16>, i64
          } else {
            scf.yield %arg20, %arg21, %arg22, %arg23, %arg27, %arg28, %arg30, %arg31 : tensor<16x2x16x16xf16>, i32, i32, i64, tensor<16x2x16x16xf16>, i64, tensor<16x2x16x16xf16>, i64
          } {ssbuffer.if}
          %true_42 = arith.constant true
          %183 = arith.cmpi sgt, %170, %c0_i32_30 : i32
          %184 = arith.cmpi sgt, %165, %c0_i32_30 : i32
          %185 = arith.andi %184, %183 : i1
          %186 = arith.cmpi slt, %arg24, %152 : i64
          %187 = arith.andi %185, %186 : i1
          %188 = scf.if %187 -> (i64) {
            %201 = arith.muli %arg24, %c256_i64 : i64
            %202 = tt.splat %201 : i64 -> tensor<256xi64>
            %203 = arith.addi %202, %97 : tensor<256xi64>
            %204 = arith.cmpi slt, %203, %146 : tensor<256xi64>
            %205 = tt.expand_dims %203 {axis = 1 : i32} : tensor<256xi64> -> tensor<256x1xi64>
            %206 = tt.expand_dims %204 {axis = 1 : i32} : tensor<256xi1> -> tensor<256x1xi1>
            %207 = arith.muli %205, %cst_6 : tensor<256x1xi64>
            %208 = tt.broadcast %206 : tensor<256x1xi1> -> tensor<256x32xi1>
            %209 = arith.muli %arg24, %c256_i64 : i64
            %210 = tt.splat %209 : i64 -> tensor<256xi64>
            %211 = arith.addi %210, %97 : tensor<256xi64>
            %212 = arith.cmpi slt, %211, %146 : tensor<256xi64>
            %213 = tt.expand_dims %211 {axis = 1 : i32} : tensor<256xi64> -> tensor<256x1xi64>
            %214 = arith.muli %213, %cst_8 : tensor<256x1xi64>
            %215 = tt.expand_dims %212 {axis = 1 : i32} : tensor<256xi1> -> tensor<256x1xi1>
            %216 = tt.broadcast %215 : tensor<256x1xi1> -> tensor<256x64xi1>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 9
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 7
            %217 = tt.broadcast %214 : tensor<256x1xi64> -> tensor<256x64xi64>
            %218 = arith.addi %217, %101 : tensor<256x64xi64>
            %219 = tt.addptr %147, %218 : tensor<256x64x!tt.ptr<f16>>, tensor<256x64xi64>
            %memspacecast = memref.memory_space_cast %alloc_13 : memref<256x64xf32, #hivm.address_space<ub>> to memref<256x64xf32>
            %220 = bufferization.to_tensor %memspacecast restrict writable : memref<256x64xf32>
            %221 = arith.truncf %220 : tensor<256x64xf32> to tensor<256x64xf16>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 2
            tt.store %219, %221, %216 : tensor<256x64x!tt.ptr<f16>>
            %222 = tt.broadcast %207 : tensor<256x1xi64> -> tensor<256x32xi64>
            %223 = arith.addi %222, %103 : tensor<256x32xi64>
            %224 = tt.addptr %148, %223 : tensor<256x32x!tt.ptr<f16>>, tensor<256x32xi64>
            %memspacecast_45 = memref.memory_space_cast %alloc_18 : memref<256x32xf32, #hivm.address_space<ub>> to memref<256x32xf32>
            %225 = bufferization.to_tensor %memspacecast_45 restrict writable : memref<256x32xf32>
            %226 = arith.truncf %225 : tensor<256x32xf32> to tensor<256x32xf16>
            tt.store %224, %226, %208 : tensor<256x32x!tt.ptr<f16>>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
            %227 = llvm.load %62 : !llvm.ptr<11> -> i32
            %228 = arith.subi %227, %c1_i32_31 : i32
            llvm.store %228, %62 : i32, !llvm.ptr<11>
            %229 = llvm.load %52 : !llvm.ptr<11> -> i32
            %230 = arith.subi %229, %c1_i32_31 : i32
            llvm.store %230, %52 : i32, !llvm.ptr<11>
            %231 = arith.addi %arg24, %c1_i64 : i64
            scf.yield %231 : i64
          } else {
            scf.yield %arg24 : i64
          } {controlValues = [], ssbuffer.if, ssbuffer.level = 1 : i32}
          %true_43 = arith.constant true
          %189 = arith.cmpi sgt, %182#1, %c0_i32_30 : i32
          %190 = arith.cmpi slt, %167, %c1_i32_25 : i32
          %191 = arith.andi %190, %189 : i1
          %192 = arith.cmpi slt, %arg25, %152 : i64
          %193 = arith.andi %191, %192 : i1
          %194:3 = scf.if %193 -> (i32, i64, i64) {
            %201 = arith.remsi %arg29, %c2_i64 : i64
            %202 = arith.cmpi eq, %201, %c0_i64_37 : i64
            %splat = tensor.splat %202 : tensor<16x2x16x16xi1>
            %203 = arith.select %splat, %182#0, %182#4 : tensor<16x2x16x16xi1>, tensor<16x2x16x16xf16>
            %204 = arith.addi %arg29, %c1_i64_38 : i64
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 5
            hivm.hir.copy ins(%203 : tensor<16x2x16x16xf16>) outs(%alloc_15 : memref<16x2x16x16xf16, #hivm.address_space<cbuf>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 5
            %205 = arith.subi %182#1, %c1_i32_31 : i32
            %206 = llvm.load %56 : !llvm.ptr<11> -> i32
            %207 = arith.addi %206, %c1_i32_31 : i32
            llvm.store %207, %56 : i32, !llvm.ptr<11>
            %208 = arith.addi %arg25, %c1_i64 : i64
            scf.yield %205, %208, %204 : i32, i64, i64
          } else {
            scf.yield %182#1, %arg25, %arg29 : i32, i64, i64
          } {ssbuffer.if}
          %true_44 = arith.constant true
          %195 = arith.cmpi sgt, %182#2, %c0_i32_30 : i32
          %196 = arith.cmpi slt, %166, %c1_i32_24 : i32
          %197 = arith.andi %196, %195 : i1
          %198 = arith.cmpi slt, %arg26, %152 : i64
          %199 = arith.andi %197, %198 : i1
          %200:3 = scf.if %199 -> (i32, i64, i64) {
            %201 = arith.remsi %arg32, %c2_i64_41 : i64
            %202 = arith.cmpi eq, %201, %c0_i64_39 : i64
            %splat = tensor.splat %202 : tensor<16x2x16x16xi1>
            %203 = arith.select %splat, %182#0, %182#6 : tensor<16x2x16x16xi1>, tensor<16x2x16x16xf16>
            %204 = arith.addi %arg32, %c1_i64_40 : i64
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 6
            hivm.hir.copy ins(%203 : tensor<16x2x16x16xf16>) outs(%alloc_14 : memref<16x2x16x16xf16, #hivm.address_space<cbuf>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 6
            %205 = arith.subi %182#2, %c1_i32_31 : i32
            %206 = llvm.load %54 : !llvm.ptr<11> -> i32
            %207 = arith.addi %206, %c1_i32_31 : i32
            llvm.store %207, %54 : i32, !llvm.ptr<11>
            %208 = arith.addi %arg26, %c1_i64 : i64
            scf.yield %205, %208, %204 : i32, i64, i64
          } else {
            scf.yield %182#2, %arg26, %arg32 : i32, i64, i64
          } {ssbuffer.if}
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 14
          scf.yield %182#0, %194#0, %200#0, %182#3, %188, %194#1, %200#1, %182#4, %182#5, %194#2, %182#6, %182#7, %200#2 : tensor<16x2x16x16xf16>, i32, i32, i64, i64, i64, i64, tensor<16x2x16x16xf16>, i64, i64, tensor<16x2x16x16xf16>, i64, i64
        } {ssbuffer.mainloop}
        %154 = tt.splat %131 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
        %155 = tt.addptr %154, %142 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi64>
        %156 = tt.broadcast %155 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x64x!tt.ptr<f32>>
        %157 = tt.addptr %156, %95 : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
        %158 = tt.load %157, %143, %cst_11 : tensor<32x64x!tt.ptr<f32>>
        %159 = arith.truncf %158 : tensor<32x64xf32> to tensor<32x64xf16>
        %160 = tt.splat %132 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>>
        %161 = tt.addptr %160, %142 : tensor<32x1x!tt.ptr<f16>>, tensor<32x1xi64>
        %162 = tt.broadcast %161 : tensor<32x1x!tt.ptr<f16>> -> tensor<32x64x!tt.ptr<f16>>
        %163 = tt.addptr %162, %95 : tensor<32x64x!tt.ptr<f16>>, tensor<32x64xi32>
        tt.store %163, %159, %143 : tensor<32x64x!tt.ptr<f16>>
        scf.yield %111#1, %111#0, %111#2 : i64, i64, i64
      }
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 8
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 5
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 6
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %c1_i32_22 = arith.constant 1 : i32
      %c1_i32_23 = arith.constant 1 : i32
      %c1_i32_24 = arith.constant 1 : i32
      %c1_i32_25 = arith.constant 1 : i32
      %c1_i32_26 = arith.constant 1 : i32
      %c1_i32_27 = arith.constant 1 : i32
      %c1_i32_28 = arith.constant 1 : i32
      %c0_i32_29 = arith.constant 0 : i32
      %c1_i32_30 = arith.constant 1 : i32
      %c24_i64 = arith.constant 24 : i64
      %c1048_i64 = arith.constant 1048 : i64
      %39 = llvm.inttoptr %c24_i64 : i64 to !llvm.ptr<11>
      %40 = llvm.inttoptr %c1048_i64 : i64 to !llvm.ptr<11>
      %c20_i64 = arith.constant 20 : i64
      %c1044_i64 = arith.constant 1044 : i64
      %41 = llvm.inttoptr %c20_i64 : i64 to !llvm.ptr<11>
      %42 = llvm.inttoptr %c1044_i64 : i64 to !llvm.ptr<11>
      %c16_i64 = arith.constant 16 : i64
      %c1040_i64 = arith.constant 1040 : i64
      %43 = llvm.inttoptr %c16_i64 : i64 to !llvm.ptr<11>
      %44 = llvm.inttoptr %c1040_i64 : i64 to !llvm.ptr<11>
      %c12_i64 = arith.constant 12 : i64
      %c1036_i64 = arith.constant 1036 : i64
      %45 = llvm.inttoptr %c12_i64 : i64 to !llvm.ptr<11>
      %46 = llvm.inttoptr %c1036_i64 : i64 to !llvm.ptr<11>
      %c8_i64 = arith.constant 8 : i64
      %c1032_i64 = arith.constant 1032 : i64
      %47 = llvm.inttoptr %c8_i64 : i64 to !llvm.ptr<11>
      %48 = llvm.inttoptr %c1032_i64 : i64 to !llvm.ptr<11>
      %c4_i64_31 = arith.constant 4 : i64
      %c1028_i64 = arith.constant 1028 : i64
      %49 = llvm.inttoptr %c4_i64_31 : i64 to !llvm.ptr<11>
      %50 = llvm.inttoptr %c1028_i64 : i64 to !llvm.ptr<11>
      %c0_i64_32 = arith.constant 0 : i64
      %c1024_i64 = arith.constant 1024 : i64
      %51 = llvm.inttoptr %c0_i64_32 : i64 to !llvm.ptr<11>
      %52 = llvm.inttoptr %c1024_i64 : i64 to !llvm.ptr<11>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 6
      hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 5
      hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 8
      %53 = tt.get_num_programs x : i32
      %54 = scf.for %arg15 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg16 = %c0_i64) -> (i64)  : i32 {
        %88 = tt.addptr %arg10, %arg15 : !tt.ptr<i64>, i32
        %89 = tt.addptr %88, %c1_i32 : !tt.ptr<i64>, i32
        %90 = tt.load %89 : !tt.ptr<i64>
        %91 = tt.load %88 : !tt.ptr<i64>
        %92 = arith.subi %90, %91 : i64
        %93 = arith.addi %92, %c31_i64 : i64
        %94 = arith.divsi %93, %c32_i64 : i64
        %95 = arith.addi %arg16, %94 : i64
        scf.yield %95 : i64
      }
      %55 = arith.muli %54, %c4_i64 : i64
      %56 = arith.extsi %53 : i32 to i64
      %57 = arith.subi %53, %c1_i32 : i32
      %58 = arith.extsi %57 : i32 to i64
      %59 = arith.addi %55, %58 : i64
      %60 = arith.divsi %59, %56 : i64
      %61 = arith.extsi %0 : i32 to i64
      %62 = arith.muli %61, %60 : i64
      %63 = arith.addi %0, %c1_i32 : i32
      %64 = arith.extsi %63 : i32 to i64
      %65 = arith.muli %64, %60 : i64
      %66 = arith.minsi %65, %55 : i64
      %67 = arith.remsi %62, %54 : i64
      %68:3 = scf.for %arg15 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64) -> (i64, i64, i64)  : i32 {
        %88 = tt.addptr %arg10, %arg15 : !tt.ptr<i64>, i32
        %89 = tt.addptr %88, %c1_i32 : !tt.ptr<i64>, i32
        %90 = tt.load %89 : !tt.ptr<i64>
        %91 = tt.load %88 : !tt.ptr<i64>
        %92 = arith.subi %90, %91 : i64
        %93 = arith.addi %92, %c31_i64 : i64
        %94 = arith.divsi %93, %c32_i64 : i64
        %95 = arith.cmpi sge, %67, %arg18 : i64
        %96 = arith.extsi %arg15 : i32 to i64
        %97 = arith.select %95, %96, %arg16 : i64
        %98 = arith.select %95, %arg18, %arg17 : i64
        %99 = arith.addi %arg18, %94 : i64
        scf.yield %97, %98, %99 : i64, i64, i64
      }
      %69 = tt.addptr %arg10, %68#0 : !tt.ptr<i64>, i64
      %70 = tt.addptr %69, %c1_i32 : !tt.ptr<i64>, i32
      %71 = tt.load %70 : !tt.ptr<i64>
      %72 = tt.load %69 : !tt.ptr<i64>
      %73 = arith.subi %71, %72 : i64
      %74 = arith.addi %73, %c31_i64 : i64
      %75 = arith.divsi %74, %c32_i64 : i64
      %76 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %77 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
      %78 = arith.extsi %76 : tensor<32xi32> to tensor<32xi64>
      %79 = tt.expand_dims %76 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
      %80 = tt.broadcast %79 : tensor<1x32xi32> -> tensor<32x32xi32>
      %81 = tt.expand_dims %77 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %82 = tt.broadcast %81 : tensor<1x64xi32> -> tensor<32x64xi32>
      %83 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %84 = arith.extsi %83 : tensor<256xi32> to tensor<256xi64>
      %85 = tt.broadcast %81 : tensor<1x64xi32> -> tensor<256x64xi32>
      %86 = tt.broadcast %79 : tensor<1x32xi32> -> tensor<256x32xi32>
      %87:3 = scf.for %arg15 = %62 to %66 step %c1_i64 iter_args(%arg16 = %68#0, %arg17 = %68#1, %arg18 = %75) -> (i64, i64, i64)  : i64 {
        %88 = arith.divsi %arg15, %54 : i64
        %89 = arith.remsi %arg15, %54 : i64
        %90 = arith.cmpi slt, %89, %arg17 : i64
        %91 = arith.select %90, %c0_i64, %arg16 : i64
        %92 = arith.select %90, %c0_i64, %arg17 : i64
        %93 = scf.if %90 -> (i64) {
          %156 = tt.addptr %arg10, %c1_i32 : !tt.ptr<i64>, i32
          %157 = tt.load %156 : !tt.ptr<i64>
          %158 = tt.load %arg10 : !tt.ptr<i64>
          %159 = arith.subi %157, %158 : i64
          %160 = arith.addi %159, %c31_i64 : i64
          %161 = arith.divsi %160, %c32_i64 : i64
          scf.yield %161 : i64
        } else {
          scf.yield %arg18 : i64
        }
        %94:3 = scf.while (%arg19 = %92, %arg20 = %91, %arg21 = %93) : (i64, i64, i64) -> (i64, i64, i64) {
          %156 = arith.addi %arg19, %arg21 : i64
          %157 = arith.cmpi sge, %89, %156 : i64
          scf.condition(%157) %arg19, %arg20, %arg21 : i64, i64, i64
        } do {
        ^bb0(%arg19: i64, %arg20: i64, %arg21: i64):
          %156 = arith.addi %arg19, %arg21 : i64
          %157 = arith.addi %arg20, %c1_i64 : i64
          %158 = tt.addptr %arg10, %157 : !tt.ptr<i64>, i64
          %159 = tt.addptr %158, %c1_i32 : !tt.ptr<i64>, i32
          %160 = tt.load %159 : !tt.ptr<i64>
          %161 = tt.load %158 : !tt.ptr<i64>
          %162 = arith.subi %160, %161 : i64
          tt.assert %true, "int32 overflow detected for operation sub" : i1
          %163 = arith.addi %162, %c31_i64 : i64
          %164 = arith.divsi %163, %c32_i64 : i64
          scf.yield %156, %157, %164 : i64, i64, i64
        }
        %95 = arith.subi %89, %94#0 : i64
        %96 = tt.addptr %arg10, %94#1 : !tt.ptr<i64>, i64
        %97 = tt.load %96 : !tt.ptr<i64>
        %98 = tt.addptr %96, %c1_i32 : !tt.ptr<i64>, i32
        %99 = tt.load %98 : !tt.ptr<i64>
        %100 = tt.addptr %arg11, %94#1 : !tt.ptr<i64>, i64
        %101 = tt.load %100 : !tt.ptr<i64>
        %102 = tt.addptr %100, %c1_i32 : !tt.ptr<i64>, i32
        %103 = tt.load %102 : !tt.ptr<i64>
        %104 = arith.subi %99, %97 : i64
        %105 = arith.subi %103, %101 : i64
        %106 = arith.muli %88, %c64_i64 : i64
        %107 = arith.muli %97, %c256_i64 : i64
        %108 = arith.addi %106, %107 : i64
        %109 = tt.addptr %arg1, %108 : !tt.ptr<f16>, i64
        %110 = arith.muli %101, %c256_i64 : i64
        %111 = arith.addi %106, %110 : i64
        %112 = tt.addptr %arg2, %111 : !tt.ptr<f16>, i64
        %113 = arith.muli %88, %c32_i64 : i64
        %114 = arith.muli %101, %c128_i64 : i64
        %115 = arith.addi %113, %114 : i64
        %116 = tt.addptr %arg3, %115 : !tt.ptr<f16>, i64
        %117 = arith.muli %97, %c128_i64 : i64
        %118 = arith.addi %113, %117 : i64
        %119 = tt.addptr %arg0, %118 : !tt.ptr<f16>, i64
        %120 = tt.addptr %arg4, %108 : !tt.ptr<f32>, i64
        %121 = arith.muli %95, %c32_i64 : i64
        %122 = tt.splat %121 : i64 -> tensor<32xi64>
        %123 = arith.addi %122, %78 : tensor<32xi64>
        %124 = tt.splat %104 : i64 -> tensor<32xi64>
        %125 = arith.cmpi slt, %123, %124 : tensor<32xi64>
        %126 = tt.expand_dims %123 {axis = 1 : i32} : tensor<32xi64> -> tensor<32x1xi64>
        %127 = arith.muli %126, %cst_0 : tensor<32x1xi64>
        %128 = tt.splat %119 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>>
        %129 = tt.addptr %128, %127 : tensor<32x1x!tt.ptr<f16>>, tensor<32x1xi64>
        %130 = tt.broadcast %129 : tensor<32x1x!tt.ptr<f16>> -> tensor<32x32x!tt.ptr<f16>>
        %131 = tt.addptr %130, %80 : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
        %132 = tt.expand_dims %125 {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
        %133 = tt.broadcast %132 : tensor<32x1xi1> -> tensor<32x32xi1>
        %134 = tt.load %131, %133, %cst_2 : tensor<32x32x!tt.ptr<f16>>
        %135 = arith.muli %126, %cst_1 : tensor<32x1xi64>
        %136 = tt.splat %109 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>>
        %137 = tt.addptr %136, %135 : tensor<32x1x!tt.ptr<f16>>, tensor<32x1xi64>
        %138 = tt.broadcast %137 : tensor<32x1x!tt.ptr<f16>> -> tensor<32x64x!tt.ptr<f16>>
        %139 = tt.addptr %138, %82 : tensor<32x64x!tt.ptr<f16>>, tensor<32x64xi32>
        %140 = tt.broadcast %132 : tensor<32x1xi1> -> tensor<32x64xi1>
        %141 = tt.load %139, %140, %cst_3 : tensor<32x64x!tt.ptr<f16>>
        %142 = arith.addi %105, %c255_i64 : i64
        %143 = arith.divsi %142, %c256_i64 : i64
        %144 = tt.splat %105 : i64 -> tensor<256xi64>
        %145 = tt.splat %112 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>>
        %146 = tt.splat %116 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>>
        %147 = tt.splat %120 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
        %148 = tt.addptr %147, %135 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi64>
        %149 = tt.broadcast %148 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x64x!tt.ptr<f32>>
        %150 = tt.addptr %149, %82 : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
        %c4_i64_33 = arith.constant 4 : i64
        %151 = arith.muli %c1_i64, %c4_i64_33 : i64
        %152 = arith.addi %143, %151 : i64
        %c4_i64_34 = arith.constant 4 : i64
        %153 = arith.muli %c1_i64, %c4_i64_34 : i64
        %154 = arith.subi %152, %153 : i64
        %155:4 = scf.for %arg19 = %c0_i64 to %152 step %c1_i64 iter_args(%arg20 = %c0_i64, %arg21 = %c0_i64, %arg22 = %c0_i64, %arg23 = %c0_i64) -> (i64, i64, i64, i64)  : i64 {
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 14
          %156 = llvm.load %39 : !llvm.ptr<11> -> i32
          %157 = llvm.load %40 : !llvm.ptr<11> -> i32
          %158 = llvm.load %41 : !llvm.ptr<11> -> i32
          %159 = llvm.load %42 : !llvm.ptr<11> -> i32
          %160 = llvm.load %43 : !llvm.ptr<11> -> i32
          %161 = llvm.load %44 : !llvm.ptr<11> -> i32
          %162 = llvm.load %45 : !llvm.ptr<11> -> i32
          %163 = llvm.load %46 : !llvm.ptr<11> -> i32
          %164 = llvm.load %47 : !llvm.ptr<11> -> i32
          %165 = llvm.load %48 : !llvm.ptr<11> -> i32
          %166 = llvm.load %49 : !llvm.ptr<11> -> i32
          %167 = llvm.load %50 : !llvm.ptr<11> -> i32
          %168 = llvm.load %51 : !llvm.ptr<11> -> i32
          %169 = llvm.load %52 : !llvm.ptr<11> -> i32
          %true_35 = arith.constant true
          %170 = arith.cmpi slt, %158, %c1_i32_27 : i32
          %171 = arith.cmpi slt, %159, %c1_i32_27 : i32
          %172 = arith.andi %170, %171 : i1
          %173 = arith.cmpi slt, %arg20, %154 : i64
          %174 = arith.andi %172, %173 : i1
          %175 = scf.if %174 -> (i64) {
            %206 = arith.muli %arg20, %c256_i64 : i64
            %207 = tt.splat %206 : i64 -> tensor<256xi64>
            %208 = arith.addi %207, %84 : tensor<256xi64>
            %209 = arith.cmpi slt, %208, %144 : tensor<256xi64>
            %210 = tt.expand_dims %208 {axis = 1 : i32} : tensor<256xi64> -> tensor<256x1xi64>
            %211 = arith.muli %210, %cst_8 : tensor<256x1xi64>
            %212 = tt.addptr %145, %211 : tensor<256x1x!tt.ptr<f16>>, tensor<256x1xi64>
            %213 = tt.broadcast %212 : tensor<256x1x!tt.ptr<f16>> -> tensor<256x64x!tt.ptr<f16>>
            %214 = tt.addptr %213, %85 : tensor<256x64x!tt.ptr<f16>>, tensor<256x64xi32>
            %215 = tt.expand_dims %209 {axis = 1 : i32} : tensor<256xi1> -> tensor<256x1xi1>
            %216 = tt.broadcast %215 : tensor<256x1xi1> -> tensor<256x64xi1>
            %217 = tt.load %214, %216, %cst_5 : tensor<256x64x!tt.ptr<f16>>
            %218 = tt.trans %217 {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
            %219 = tt.dot %141, %218, %cst_7 : tensor<32x64xf16> * tensor<64x256xf16> -> tensor<32x256xf32>
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%219 : tensor<32x256xf32>) outs(%alloc_17 : memref<32x256xf32, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
            %220 = llvm.load %41 : !llvm.ptr<11> -> i32
            %221 = llvm.load %42 : !llvm.ptr<11> -> i32
            %222 = arith.addi %220, %c1_i32_30 : i32
            %223 = arith.addi %221, %c1_i32_30 : i32
            llvm.store %222, %41 : i32, !llvm.ptr<11>
            llvm.store %223, %42 : i32, !llvm.ptr<11>
            %224 = arith.addi %arg20, %c1_i64 : i64
            scf.yield %224 : i64
          } else {
            scf.yield %arg20 : i64
          } {controlValues = [], ssbuffer.if}
          %true_36 = arith.constant true
          %176 = arith.cmpi slt, %160, %c1_i32_26 : i32
          %177 = arith.cmpi slt, %161, %c1_i32_26 : i32
          %178 = arith.andi %176, %177 : i1
          %179 = arith.cmpi slt, %arg21, %154 : i64
          %180 = arith.andi %178, %179 : i1
          %181 = scf.if %180 -> (i64) {
            %206 = arith.muli %arg21, %c256_i64 : i64
            %207 = tt.splat %206 : i64 -> tensor<256xi64>
            %208 = arith.addi %207, %84 : tensor<256xi64>
            %209 = arith.cmpi slt, %208, %144 : tensor<256xi64>
            %210 = tt.expand_dims %208 {axis = 1 : i32} : tensor<256xi64> -> tensor<256x1xi64>
            %211 = arith.muli %210, %cst_8 : tensor<256x1xi64>
            %212 = tt.addptr %145, %211 : tensor<256x1x!tt.ptr<f16>>, tensor<256x1xi64>
            %213 = tt.broadcast %212 : tensor<256x1x!tt.ptr<f16>> -> tensor<256x64x!tt.ptr<f16>>
            %214 = tt.addptr %213, %85 : tensor<256x64x!tt.ptr<f16>>, tensor<256x64xi32>
            %215 = tt.expand_dims %209 {axis = 1 : i32} : tensor<256xi1> -> tensor<256x1xi1>
            %216 = tt.broadcast %215 : tensor<256x1xi1> -> tensor<256x64xi1>
            %217 = tt.load %214, %216, %cst_5 : tensor<256x64x!tt.ptr<f16>>
            %218 = tt.trans %217 {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
            %219 = arith.muli %210, %cst_6 : tensor<256x1xi64>
            %220 = tt.addptr %146, %219 : tensor<256x1x!tt.ptr<f16>>, tensor<256x1xi64>
            %221 = tt.broadcast %220 : tensor<256x1x!tt.ptr<f16>> -> tensor<256x32x!tt.ptr<f16>>
            %222 = tt.addptr %221, %86 : tensor<256x32x!tt.ptr<f16>>, tensor<256x32xi32>
            %223 = tt.broadcast %215 : tensor<256x1xi1> -> tensor<256x32xi1>
            %224 = tt.load %222, %223, %cst_4 : tensor<256x32x!tt.ptr<f16>>
            %225 = tt.trans %224 {order = array<i32: 1, 0>} : tensor<256x32xf16> -> tensor<32x256xf16>
            %226 = tt.dot %134, %225, %cst_7 : tensor<32x32xf16> * tensor<32x256xf16> -> tensor<32x256xf32>
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%226 : tensor<32x256xf32>) outs(%alloc_16 : memref<32x256xf32, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 4
            %227 = llvm.load %43 : !llvm.ptr<11> -> i32
            %228 = llvm.load %44 : !llvm.ptr<11> -> i32
            %229 = arith.addi %227, %c1_i32_30 : i32
            %230 = arith.addi %228, %c1_i32_30 : i32
            llvm.store %229, %43 : i32, !llvm.ptr<11>
            llvm.store %230, %44 : i32, !llvm.ptr<11>
            %231 = arith.addi %arg21, %c1_i64 : i64
            scf.yield %231 : i64
          } else {
            scf.yield %arg21 : i64
          } {controlValues = [], ssbuffer.if}
          %true_37 = arith.constant true
          %182 = arith.cmpi sgt, %164, %c0_i32_29 : i32
          %183 = arith.cmpi sgt, %165, %c0_i32_29 : i32
          %184 = arith.andi %182, %182 : i1
          %185 = arith.cmpi sgt, %162, %c0_i32_29 : i32
          %186 = arith.cmpi sgt, %163, %c0_i32_29 : i32
          %187 = arith.andi %185, %185 : i1
          %188 = arith.andi %184, %187 : i1
          %189 = arith.cmpi slt, %166, %c1_i32_23 : i32
          %190 = arith.cmpi slt, %167, %c1_i32_23 : i32
          %191 = arith.andi %189, %190 : i1
          %192 = arith.andi %188, %191 : i1
          %193 = arith.cmpi slt, %arg22, %154 : i64
          %194 = arith.andi %192, %193 : i1
          %195 = scf.if %194 -> (i64) {
            %206 = arith.muli %arg22, %c256_i64 : i64
            %207 = tt.splat %206 : i64 -> tensor<256xi64>
            %208 = arith.addi %207, %84 : tensor<256xi64>
            %209 = arith.cmpi slt, %208, %144 : tensor<256xi64>
            %210 = tt.expand_dims %208 {axis = 1 : i32} : tensor<256xi64> -> tensor<256x1xi64>
            %211 = arith.muli %210, %cst_8 : tensor<256x1xi64>
            %212 = tt.addptr %145, %211 : tensor<256x1x!tt.ptr<f16>>, tensor<256x1xi64>
            %213 = tt.broadcast %212 : tensor<256x1x!tt.ptr<f16>> -> tensor<256x64x!tt.ptr<f16>>
            %214 = tt.addptr %213, %85 : tensor<256x64x!tt.ptr<f16>>, tensor<256x64xi32>
            %215 = tt.expand_dims %209 {axis = 1 : i32} : tensor<256xi1> -> tensor<256x1xi1>
            %216 = tt.broadcast %215 : tensor<256x1xi1> -> tensor<256x64xi1>
            %217 = tt.load %214, %216, %cst_5 : tensor<256x64x!tt.ptr<f16>>
            %218 = tt.trans %217 {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
            %219 = arith.muli %210, %cst_6 : tensor<256x1xi64>
            %220 = tt.addptr %146, %219 : tensor<256x1x!tt.ptr<f16>>, tensor<256x1xi64>
            %221 = tt.broadcast %220 : tensor<256x1x!tt.ptr<f16>> -> tensor<256x32x!tt.ptr<f16>>
            %222 = tt.addptr %221, %86 : tensor<256x32x!tt.ptr<f16>>, tensor<256x32xi32>
            %223 = tt.broadcast %215 : tensor<256x1xi1> -> tensor<256x32xi1>
            %224 = tt.load %222, %223, %cst_4 : tensor<256x32x!tt.ptr<f16>>
            %225 = tt.trans %224 {order = array<i32: 1, 0>} : tensor<256x32xf16> -> tensor<32x256xf16>
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 5
            %226 = hivm.hir.convert_layout %alloc_15 output_shape [32, 256] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<16x2x16x16xf16, #hivm.address_space<cbuf>>) -> memref<32x256xf16, #hivm.address_space<cbuf>>
            %memspacecast = memref.memory_space_cast %226 : memref<32x256xf16, #hivm.address_space<cbuf>> to memref<32x256xf16>
            %227 = bufferization.to_tensor %memspacecast restrict writable : memref<32x256xf16>
            %228 = tt.dot %227, %217, %cst_11 : tensor<32x256xf16> * tensor<256x64xf16> -> tensor<32x64xf32>
            %229 = tt.atomic_rmw fadd, acq_rel, gpu, %150, %228, %140 : (tensor<32x64x!tt.ptr<f32>>, tensor<32x64xf32>, tensor<32x64xi1>) -> tensor<32x64xf32>
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 2
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 6
            %230 = hivm.hir.convert_layout %alloc_14 output_shape [32, 256] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<16x2x16x16xf16, #hivm.address_space<cbuf>>) -> memref<32x256xf16, #hivm.address_space<cbuf>>
            %memspacecast_39 = memref.memory_space_cast %230 : memref<32x256xf16, #hivm.address_space<cbuf>> to memref<32x256xf16>
            %231 = bufferization.to_tensor %memspacecast_39 restrict writable : memref<32x256xf16>
            %232 = tt.trans %231 {order = array<i32: 1, 0>} : tensor<32x256xf16> -> tensor<256x32xf16>
            %233 = tt.dot %232, %141, %cst_12 : tensor<256x32xf16> * tensor<32x64xf16> -> tensor<256x64xf32>
            hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 5
            hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 6
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 7
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%233 : tensor<256x64xf32>) outs(%alloc_13 : memref<256x64xf32, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 7
            %234 = llvm.load %47 : !llvm.ptr<11> -> i32
            %235 = llvm.load %48 : !llvm.ptr<11> -> i32
            %236 = arith.subi %234, %c1_i32_30 : i32
            %237 = arith.subi %235, %c1_i32_30 : i32
            llvm.store %236, %47 : i32, !llvm.ptr<11>
            llvm.store %237, %48 : i32, !llvm.ptr<11>
            %238 = llvm.load %45 : !llvm.ptr<11> -> i32
            %239 = llvm.load %46 : !llvm.ptr<11> -> i32
            %240 = arith.subi %238, %c1_i32_30 : i32
            %241 = arith.subi %239, %c1_i32_30 : i32
            llvm.store %240, %45 : i32, !llvm.ptr<11>
            llvm.store %241, %46 : i32, !llvm.ptr<11>
            %242 = llvm.load %49 : !llvm.ptr<11> -> i32
            %243 = llvm.load %50 : !llvm.ptr<11> -> i32
            %244 = arith.addi %242, %c1_i32_30 : i32
            %245 = arith.addi %243, %c1_i32_30 : i32
            llvm.store %244, %49 : i32, !llvm.ptr<11>
            llvm.store %245, %50 : i32, !llvm.ptr<11>
            %246 = arith.addi %arg22, %c1_i64 : i64
            scf.yield %246 : i64
          } else {
            scf.yield %arg22 : i64
          } {controlValues = [], ssbuffer.if}
          %true_38 = arith.constant true
          %196 = arith.cmpi sgt, %168, %c0_i32_29 : i32
          %197 = arith.cmpi sgt, %169, %c0_i32_29 : i32
          %198 = arith.andi %196, %196 : i1
          %199 = arith.cmpi slt, %156, %c1_i32_28 : i32
          %200 = arith.cmpi slt, %157, %c1_i32_28 : i32
          %201 = arith.andi %199, %200 : i1
          %202 = arith.andi %198, %201 : i1
          %203 = arith.cmpi slt, %arg23, %154 : i64
          %204 = arith.andi %202, %203 : i1
          %205 = scf.if %204 -> (i64) {
            %206 = arith.muli %arg23, %c256_i64 : i64
            %207 = tt.splat %206 : i64 -> tensor<256xi64>
            %208 = arith.addi %207, %84 : tensor<256xi64>
            %209 = arith.cmpi slt, %208, %144 : tensor<256xi64>
            %210 = tt.expand_dims %208 {axis = 1 : i32} : tensor<256xi64> -> tensor<256x1xi64>
            %211 = arith.muli %210, %cst_8 : tensor<256x1xi64>
            %212 = tt.addptr %145, %211 : tensor<256x1x!tt.ptr<f16>>, tensor<256x1xi64>
            %213 = tt.broadcast %212 : tensor<256x1x!tt.ptr<f16>> -> tensor<256x64x!tt.ptr<f16>>
            %214 = tt.addptr %213, %85 : tensor<256x64x!tt.ptr<f16>>, tensor<256x64xi32>
            %215 = tt.expand_dims %209 {axis = 1 : i32} : tensor<256xi1> -> tensor<256x1xi1>
            %216 = tt.broadcast %215 : tensor<256x1xi1> -> tensor<256x64xi1>
            %217 = tt.load %214, %216, %cst_5 : tensor<256x64x!tt.ptr<f16>>
            %218 = tt.trans %217 {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
            %219 = arith.muli %210, %cst_6 : tensor<256x1xi64>
            %220 = tt.addptr %146, %219 : tensor<256x1x!tt.ptr<f16>>, tensor<256x1xi64>
            %221 = tt.broadcast %220 : tensor<256x1x!tt.ptr<f16>> -> tensor<256x32x!tt.ptr<f16>>
            %222 = tt.addptr %221, %86 : tensor<256x32x!tt.ptr<f16>>, tensor<256x32xi32>
            %223 = tt.broadcast %215 : tensor<256x1xi1> -> tensor<256x32xi1>
            %224 = tt.load %222, %223, %cst_4 : tensor<256x32x!tt.ptr<f16>>
            %225 = tt.trans %224 {order = array<i32: 1, 0>} : tensor<256x32xf16> -> tensor<32x256xf16>
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 8
            %226 = hivm.hir.convert_layout %alloc output_shape [32, 256] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<16x2x16x16xf16, #hivm.address_space<cbuf>>) -> memref<32x256xf16, #hivm.address_space<cbuf>>
            %memspacecast = memref.memory_space_cast %226 : memref<32x256xf16, #hivm.address_space<cbuf>> to memref<32x256xf16>
            %227 = bufferization.to_tensor %memspacecast restrict writable : memref<32x256xf16>
            %228 = tt.trans %227 {order = array<i32: 1, 0>} : tensor<32x256xf16> -> tensor<256x32xf16>
            %229 = tt.dot %228, %134, %cst_10 : tensor<256x32xf16> * tensor<32x32xf16> -> tensor<256x32xf32>
            hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 8
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 9
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%229 : tensor<256x32xf32>) outs(%alloc_18 : memref<256x32xf32, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 9
            %230 = llvm.load %51 : !llvm.ptr<11> -> i32
            %231 = llvm.load %52 : !llvm.ptr<11> -> i32
            %232 = arith.subi %230, %c1_i32_30 : i32
            %233 = arith.subi %231, %c1_i32_30 : i32
            llvm.store %232, %51 : i32, !llvm.ptr<11>
            llvm.store %233, %52 : i32, !llvm.ptr<11>
            %234 = llvm.load %39 : !llvm.ptr<11> -> i32
            %235 = llvm.load %40 : !llvm.ptr<11> -> i32
            %236 = arith.addi %234, %c1_i32_30 : i32
            %237 = arith.addi %235, %c1_i32_30 : i32
            llvm.store %236, %39 : i32, !llvm.ptr<11>
            llvm.store %237, %40 : i32, !llvm.ptr<11>
            %238 = arith.addi %arg23, %c1_i64 : i64
            scf.yield %238 : i64
          } else {
            scf.yield %arg23 : i64
          } {controlValues = [], ssbuffer.if}
          hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
          scf.yield %175, %181, %195, %205 : i64, i64, i64, i64
        }{ssbuffer.mainloop}
        scf.yield %94#1, %94#0, %94#2 : i64, i64, i64
      }
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 7
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 9
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 14
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    tt.return
  }
}