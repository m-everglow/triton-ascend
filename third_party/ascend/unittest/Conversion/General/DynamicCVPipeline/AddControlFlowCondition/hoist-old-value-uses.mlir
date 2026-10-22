// RUN: triton-opt --ssbuf-hoist-old-value-uses --verify-each %s | FileCheck %s
// RUN: triton-opt --ssbuf-hoist-old-value-uses --ssbuf-hoist-old-value-uses --verify-each %s | FileCheck %s

// CHECK-LABEL: func.func @basic(
// CHECK: test.id = "new"
// CHECK-NEXT: test.id = "use"
// CHECK-NEXT: test.id = "gap"
func.func @basic(%cond: i1, %init: tensor<4xf32>, %x: tensor<4xf32>, %mem: memref<f32>, %scalar: f32) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %v = scf.if %cond -> tensor<4xf32> {
      %new = arith.maximumf %old, %x {test.id = "new"} : tensor<4xf32>
      %gap = math.exp %new {test.id = "gap"} : tensor<4xf32>
      %use = arith.subf %old, %new {test.id = "use"} : tensor<4xf32>
      scf.yield %new : tensor<4xf32>
    } else {
      scf.yield %old : tensor<4xf32>
    } {ssbuffer.if = 16 : i32}
    scf.yield %v : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @late_dependency(
// CHECK: test.id = "dep"
// CHECK-NEXT: test.id = "use"
// CHECK-NEXT: test.id = "noise"
func.func @late_dependency(%cond: i1, %init: tensor<4xf32>, %x: tensor<4xf32>, %mem: memref<f32>, %scalar: f32) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %v = scf.if %cond -> tensor<4xf32> {
      %new = arith.maximumf %old, %x {test.id = "new"} : tensor<4xf32>
      %gap = math.exp %new {test.id = "gap"} : tensor<4xf32>
      %dep = arith.addf %gap, %x {test.id = "dep"} : tensor<4xf32>
      %noise = math.exp %x {test.id = "noise"} : tensor<4xf32>
      %use = arith.subf %old, %dep {test.id = "use"} : tensor<4xf32>
      scf.yield %new : tensor<4xf32>
    } else {
      scf.yield %old : tensor<4xf32>
    } {ssbuffer.if = 16 : i32}
    scf.yield %v : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @multiple_uses(
// CHECK: test.id = "new"
// CHECK-NEXT: test.id = "use"
// CHECK-NEXT: test.id = "other"
// CHECK-NEXT: test.id = "gap"
func.func @multiple_uses(%cond: i1, %init: tensor<4xf32>, %x: tensor<4xf32>, %mem: memref<f32>, %scalar: f32) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %v = scf.if %cond -> tensor<4xf32> {
      %new = arith.maximumf %old, %x {test.id = "new"} : tensor<4xf32>
      %gap = math.exp %new {test.id = "gap"} : tensor<4xf32>
      %use = arith.subf %old, %new {test.id = "use"} : tensor<4xf32>
      %other = arith.addf %old, %x {test.id = "other"} : tensor<4xf32>
      scf.yield %new : tensor<4xf32>
    } else {
      scf.yield %old : tensor<4xf32>
    } {ssbuffer.if = 16 : i32}
    scf.yield %v : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @cross_memory_op(
// CHECK: test.id = "new"
// CHECK-NEXT: test.id = "use"
// CHECK-NEXT: test.id = "gap"
// CHECK-NEXT: test.id = "barrier"
// CHECK-NEXT: test.id = "noise"
func.func @cross_memory_op(%cond: i1, %init: tensor<4xf32>, %x: tensor<4xf32>, %mem: memref<f32>, %scalar: f32) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %v = scf.if %cond -> tensor<4xf32> {
      %new = arith.maximumf %old, %x {test.id = "new"} : tensor<4xf32>
      %gap = math.exp %new {test.id = "gap"} : tensor<4xf32>
      memref.store %scalar, %mem[] {test.id = "barrier"} : memref<f32>
      %noise = math.exp %x {test.id = "noise"} : tensor<4xf32>
      %use = arith.subf %old, %new {test.id = "use"} : tensor<4xf32>
      scf.yield %new : tensor<4xf32>
    } else {
      scf.yield %old : tensor<4xf32>
    } {ssbuffer.if = 16 : i32}
    scf.yield %v : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @effectful_user(
// CHECK: test.id = "new"
// CHECK-NEXT: test.id = "store"
// CHECK-NEXT: test.id = "gap"
func.func @effectful_user(%cond: i1, %init: tensor<4xf32>, %x: tensor<4xf32>, %mem: memref<f32>, %scalar: f32, %dst: memref<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %v = scf.if %cond -> tensor<4xf32> {
      %new = arith.maximumf %old, %x {test.id = "new"} : tensor<4xf32>
      %gap = math.exp %new {test.id = "gap"} : tensor<4xf32>
      bufferization.materialize_in_destination %old in writable %dst {test.id = "store"} : (tensor<4xf32>, memref<4xf32>) -> ()
      scf.yield %new : tensor<4xf32>
    } else {
      scf.yield %old : tensor<4xf32>
    } {ssbuffer.if = 16 : i32}
    scf.yield %v : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @nested_user(
// CHECK: test.id = "new"
// CHECK-NEXT: test.id = "gap"
// CHECK: test.id = "nested"
func.func @nested_user(%cond: i1, %init: tensor<4xf32>, %x: tensor<4xf32>, %mem: memref<f32>, %scalar: f32) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %v = scf.if %cond -> tensor<4xf32> {
      %new = arith.maximumf %old, %x {test.id = "new"} : tensor<4xf32>
      %gap = math.exp %new {test.id = "gap"} : tensor<4xf32>
      scf.if %cond {
        %use = arith.subf %old, %new {test.id = "nested"} : tensor<4xf32>
      }
      scf.yield %new : tensor<4xf32>
    } else {
      scf.yield %old : tensor<4xf32>
    } {ssbuffer.if = 16 : i32}
    scf.yield %v : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @unmarked(
// CHECK: test.id = "new"
// CHECK-NEXT: test.id = "gap"
// CHECK-NEXT: test.id = "use"
func.func @unmarked(%cond: i1, %init: tensor<4xf32>, %x: tensor<4xf32>, %mem: memref<f32>, %scalar: f32) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %v = scf.if %cond -> tensor<4xf32> {
      %new = arith.maximumf %old, %x {test.id = "new"} : tensor<4xf32>
      %gap = math.exp %new {test.id = "gap"} : tensor<4xf32>
      %use = arith.subf %old, %new {test.id = "use"} : tensor<4xf32>
      scf.yield %new : tensor<4xf32>
    } else {
      scf.yield %old : tensor<4xf32>
    }
    scf.yield %v : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @branch_local_values(
// CHECK: test.id = "new"
// CHECK-NEXT: test.id = "gap"
// CHECK-NEXT: test.id = "use"
func.func @branch_local_values(%cond: i1, %init: tensor<4xf32>, %x: tensor<4xf32>, %mem: memref<f32>, %scalar: f32) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %v = scf.if %cond -> tensor<4xf32> {
      %new = arith.maximumf %old, %x {test.id = "new"} : tensor<4xf32>
      %gap = math.exp %new {test.id = "gap"} : tensor<4xf32>
      %use = arith.subf %old, %new {test.id = "use"} : tensor<4xf32>
      scf.yield %new : tensor<4xf32>
    } else {
      %else = arith.addf %old, %x : tensor<4xf32>
      scf.yield %else : tensor<4xf32>
    } {ssbuffer.if = 16 : i32}
    scf.yield %v : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @update_in_else(
// CHECK: test.id = "new"
// CHECK-NEXT: test.id = "use"
// CHECK-NEXT: test.id = "gap"
func.func @update_in_else(%cond: i1, %init: tensor<4xf32>, %x: tensor<4xf32>, %mem: memref<f32>, %scalar: f32) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %v = scf.if %cond -> tensor<4xf32> {
      scf.yield %old : tensor<4xf32>
    } else {
      %new = arith.maximumf %old, %x {test.id = "new"} : tensor<4xf32>
      %gap = math.exp %new {test.id = "gap"} : tensor<4xf32>
      %use = arith.subf %old, %new {test.id = "use"} : tensor<4xf32>
      scf.yield %new : tensor<4xf32>
    } {ssbuffer.if = 16 : i32}
    scf.yield %v : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @uses_before_update(
// CHECK: test.id = "early"
// CHECK-NEXT: test.id = "new"
// CHECK-NEXT: test.id = "use"
// CHECK-NEXT: test.id = "gap"
func.func @uses_before_update(%cond: i1, %init: tensor<4xf32>, %x: tensor<4xf32>, %mem: memref<f32>, %scalar: f32) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %v = scf.if %cond -> tensor<4xf32> {
      %early = arith.addf %old, %x {test.id = "early"} : tensor<4xf32>
      %new = arith.maximumf %old, %x {test.id = "new"} : tensor<4xf32>
      %gap = math.exp %new {test.id = "gap"} : tensor<4xf32>
      %use = arith.subf %old, %new {test.id = "use"} : tensor<4xf32>
      scf.yield %new : tensor<4xf32>
    } else {
      scf.yield %old : tensor<4xf32>
    } {ssbuffer.if = 16 : i32}
    scf.yield %v : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @tensor_reduction_gap(
// CHECK: test.id = "new"
// CHECK-NEXT: test.id = "use"
// CHECK: test.id = "gap"
func.func @tensor_reduction_gap(%cond: i1, %init: tensor<4xf32>, %x: tensor<4xf32>, %mem: memref<f32>, %scalar: f32, %seed: tensor<f32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %v = scf.if %cond -> tensor<4xf32> {
      %new = arith.maximumf %old, %x {test.id = "new"} : tensor<4xf32>
      %reduced = linalg.reduce ins(%x : tensor<4xf32>) outs(%seed : tensor<f32>) dimensions = [0] (%a: f32, %b: f32) {
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
      }
      %gap = math.exp %new {test.id = "gap"} : tensor<4xf32>
      %use = arith.subf %old, %new {test.id = "use"} : tensor<4xf32>
      scf.yield %new : tensor<4xf32>
    } else {
      scf.yield %old : tensor<4xf32>
    } {ssbuffer.if = 16 : i32}
    scf.yield %v : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @scalar_unchanged(
// CHECK: test.id = "new"
// CHECK-NEXT: test.id = "gap"
// CHECK-NEXT: test.id = "use"
func.func @scalar_unchanged(%cond: i1, %init: f32, %x: f32, %mem: memref<f32>, %scalar: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%old = %init) -> f32 {
    %v = scf.if %cond -> f32 {
      %new = arith.maximumf %old, %x {test.id = "new"} : f32
      %gap = math.exp %new {test.id = "gap"} : f32
      %use = arith.subf %old, %new {test.id = "use"} : f32
      scf.yield %new : f32
    } else {
      scf.yield %old : f32
    } {ssbuffer.if = 16 : i32}
    scf.yield %v : f32
  }
  return %r : f32
}

// CHECK-LABEL: func.func @region_capture_dependency(
// CHECK: %[[CAP:.*]] = tensor.extract
// CHECK-NEXT: linalg.reduce
// CHECK: arith.addf {{.*}}, %[[CAP]]
// CHECK: test.id = "noise"
func.func @region_capture_dependency(%cond: i1, %old: tensor<4xf32>, %x: tensor<4xf32>, %seed: tensor<f32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %r = scf.if %cond -> tensor<4xf32> {
    %new = arith.maximumf %old, %x : tensor<4xf32>
    %gap = math.exp %new : tensor<4xf32>
    %capture = tensor.extract %gap[%c0] : tensor<4xf32>
    %noise = math.exp %x {test.id = "noise"} : tensor<4xf32>
    %reduced = linalg.reduce ins(%old : tensor<4xf32>) outs(%seed : tensor<f32>) dimensions = [0] (%a: f32, %b: f32) {
      %sum = arith.addf %a, %capture : f32
      linalg.yield %sum : f32
    }
    scf.yield %new : tensor<4xf32>
  } else {
    scf.yield %old : tensor<4xf32>
  } {ssbuffer.if = 16 : i32}
  return %r : tensor<4xf32>
}
