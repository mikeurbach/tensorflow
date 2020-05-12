// RUN: dataflow-opt -dataflow-legalize-hlo %s | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: needsForkOp
func @needsForkOp(%arg0: tensor<i32>) {
  // CHECK-NEXT: %0 = "dataflow.unit_rate"() ( {
  // CHECK-NEXT:   %3 = xla_hlo.constant dense<1> : tensor<i32>
  // CHECK-NEXT:   "dataflow.return"(%3) : (tensor<i32>) -> ()
  // CHECK-NEXT: }) : () -> tensor<i32>
  %0 = xla_hlo.constant dense<1> : tensor<i32>
  // CHECK-NEXT: %1 = "dataflow.fork"(%0) : (tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: %2 = "dataflow.unit_rate"(%1, %1) ( {
  // CHECK-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>): // no predecessors
  // CHECK-NEXT:   %3 = xla_hlo.add %arg1, %arg2 : tensor<i32>
  // CHECK-NEXT:   "dataflow.return"(%3) : (tensor<i32>) -> ()
  // CHECK-NEXT: }) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "xla_hlo.add"(%0, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return
}
