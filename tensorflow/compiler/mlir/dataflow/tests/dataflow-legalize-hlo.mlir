// RUN: dataflow-opt -dataflow-legalize-hlo %s | FileCheck %s --dump-input-on-failure

//===----------------------------------------------------------------------===//
// UnitRate op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: notOp
func @notOp(%arg0: tensor<i1>) {
  // CHECK-NEXT: "dataflow.unit_rate"(%arg0) ( {
  // CHECK-NEXT:   %1 = "xla_hlo.not"(%arg0) : (tensor<i1>) -> tensor<i1>
  // CHECK-NEXT:   "xla_hlo.return"(%1) : (tensor<i1>) -> ()
  // CHECK-NEXT: }) : (tensor<i1>) -> tensor<i1>
  %0 = "xla_hlo.not"(%arg0) : (tensor<i1>) -> tensor<i1>
  return
}

// CHECK-LABEL: addOp
func @addOp(%arg0: tensor<i32>, %arg1: tensor<i32>) {
  // CHECK-NEXT: "dataflow.unit_rate"(%arg0, %arg1) ( {
  // CHECK-NEXT:   %1 = xla_hlo.add %arg0, %arg1 : tensor<i32>
  // CHECK-NEXT:   "xla_hlo.return"(%1) : (tensor<i32>) -> ()
  // CHECK-NEXT: }) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return
}

// CHECK-LABEL: compareOp
func @compareOp(%arg0: tensor<i32>, %arg1: tensor<i32>) {
  // CHECK-NEXT: "dataflow.unit_rate"(%arg0, %arg1) ( {
  // CHECK-NEXT:   %1 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "NE"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT:   "xla_hlo.return"(%1) : (tensor<i1>) -> ()
  // CHECK-NEXT: }) {comparison_direction = "NE"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %0 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "NE"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return
}

//===----------------------------------------------------------------------===//
// Mux op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: selectOp
func @selectOp(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) {
  // CHECK-NEXT: "dataflow.mux"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  return
}
