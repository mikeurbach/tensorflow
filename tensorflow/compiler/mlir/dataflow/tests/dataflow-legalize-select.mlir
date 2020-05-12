// RUN: dataflow-opt -dataflow-legalize-hlo %s | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: selectOp
func @selectOp(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) {
  // CHECK-NEXT: %0 = "dataflow.mux"(%arg0, %arg2, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  return
}
