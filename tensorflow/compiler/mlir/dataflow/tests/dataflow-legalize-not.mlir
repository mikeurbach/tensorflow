// RUN: dataflow-opt -dataflow-legalize-hlo %s | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: notOp
func @notOp(%arg0: tensor<i1>) {
  // CHECK-NEXT: %0 = "dataflow.unit_rate"(%arg0) ( {
  // CHECK-NEXT: ^bb0(%arg1: tensor<i1>): // no predecessors
  // CHECK-NEXT:   %1 = "xla_hlo.not"(%arg1) : (tensor<i1>) -> tensor<i1>
  // CHECK-NEXT:   "dataflow.return"(%1) : (tensor<i1>) -> ()
  // CHECK-NEXT: }) : (tensor<i1>) -> tensor<i1>
  %0 = "xla_hlo.not"(%arg0) : (tensor<i1>) -> tensor<i1>
  return
}
