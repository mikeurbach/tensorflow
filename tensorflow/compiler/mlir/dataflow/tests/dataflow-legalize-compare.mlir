// RUN: dataflow-opt -dataflow-legalize-hlo %s | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: compareOp
func @compareOp(%arg0: tensor<i32>, %arg1: tensor<i32>) {
  // CHECK-NEXT: %0 = "dataflow.unit_rate"(%arg0, %arg1) ( {
  // CHECK-NEXT: ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>): // no predecessors
  // CHECK-NEXT:   %1 = "xla_hlo.compare"(%arg2, %arg3) {comparison_direction = "NE"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT:   "dataflow.return"(%1) : (tensor<i1>) -> ()
  // CHECK-NEXT: }) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %0 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "NE"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return
}
