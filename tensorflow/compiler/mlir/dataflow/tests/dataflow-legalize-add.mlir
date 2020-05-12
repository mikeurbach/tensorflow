// RUN: dataflow-opt -dataflow-legalize-hlo %s | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: addOp
func @addOp(%arg0: tensor<i32>, %arg1: tensor<i32>) {
  // CHECK-NEXT: %0 = "dataflow.unit_rate"(%arg0, %arg1) ( {
  // CHECK-NEXT: ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>): // no predecessors
  // CHECK-NEXT:   %1 = xla_hlo.add %arg2, %arg3 : tensor<i32>
  // CHECK-NEXT:   "dataflow.return"(%1) : (tensor<i32>) -> ()
  // CHECK-NEXT: }) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return
}
