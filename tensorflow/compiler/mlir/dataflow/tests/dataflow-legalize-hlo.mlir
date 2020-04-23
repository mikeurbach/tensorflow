// RUN: dataflow-opt -dataflow-legalize-hlo %s | FileCheck %s --dump-input-on-failure

//===----------------------------------------------------------------------===//
// UnitRate op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: notOp
func @notOp(%arg0: tensor<i1>) {
  // CHECK-NEXT: %0 = "dataflow.unit_rate"() ( {
  // CHECK-NEXT:   %1 = "xla_hlo.not"(%arg0) : (tensor<i1>) -> tensor<i1>
  // CHECK-NEXT:   "dataflow.return"(%1) : (tensor<i1>) -> ()
  // CHECK-NEXT: }) : () -> tensor<i1>
  %0 = "xla_hlo.not"(%arg0) : (tensor<i1>) -> tensor<i1>
  return
}

// CHECK-LABEL: addOp
func @addOp(%arg0: tensor<i32>, %arg1: tensor<i32>) {
  // CHECK-NEXT: %0 = "dataflow.unit_rate"() ( {
  // CHECK-NEXT:   %1 = xla_hlo.add %arg0, %arg1 : tensor<i32>
  // CHECK-NEXT:   "dataflow.return"(%1) : (tensor<i32>) -> ()
  // CHECK-NEXT: }) : () -> tensor<i32>
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return
}

// CHECK-LABEL: compareOp
func @compareOp(%arg0: tensor<i32>, %arg1: tensor<i32>) {
  // CHECK-NEXT: %0 = "dataflow.unit_rate"() ( {
  // CHECK-NEXT:   %1 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "NE"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT:   "dataflow.return"(%1) : (tensor<i1>) -> ()
  // CHECK-NEXT: }) : () -> tensor<i1>
  %0 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "NE"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return
}

//===----------------------------------------------------------------------===//
// Mux op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: selectOp
func @selectOp(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) {
  // CHECK-NEXT: %0 = "dataflow.mux"(%arg0, %arg2, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  return
}

//===----------------------------------------------------------------------===//
// Loop op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: whileOp
func @whileOp(%arg0: tensor<i32>) {
  // CHECK-NEXT: %0 = "dataflow.loop"(%arg0) ( {
  // CHECK-NEXT: ^bb0(%arg1: tensor<i1>, %arg2: tensor<i32>): // no predecessors
  // CHECK-NEXT:   %1 = "dataflow.initial"(%arg1) {value = 0 : i1} : (tensor<i1>) -> tensor<i1>
  // CHECK-NEXT:   %2 = "dataflow.mux"(%1, %arg0, %arg2) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT:   %3 = "dataflow.fork"(%2) : (tensor<i32>) -> tensor<i32>
  // CHECK-NEXT:   %4 = "dataflow.unit_rate"() ( {
  // CHECK-NEXT:     %13 = xla_hlo.constant dense<10> : tensor<i32>
  // CHECK-NEXT:     "dataflow.return"(%13) : (tensor<i32>) -> ()
  // CHECK-NEXT:   }) : () -> tensor<i32>
  // CHECK-NEXT:   %5 = "dataflow.unit_rate"() ( {
  // CHECK-NEXT:     %13 = "xla_hlo.compare"(%3, %4) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT:     "dataflow.return"(%13) : (tensor<i1>) -> ()
  // CHECK-NEXT:   }) : () -> tensor<i1>
  // CHECK-NEXT:   %6 = "dataflow.fork"(%5) : (tensor<i1>) -> tensor<i1>
  // CHECK-NEXT:   %7 = "dataflow.unit_rate"() ( {
  // CHECK-NEXT:     %13 = xla_hlo.constant dense<1> : tensor<i32>
  // CHECK-NEXT:     "dataflow.return"(%13) : (tensor<i32>) -> ()
  // CHECK-NEXT:   }) : () -> tensor<i32>
  // CHECK-NEXT:   %8 = "dataflow.unit_rate"() ( {
  // CHECK-NEXT:     %13 = xla_hlo.add %3, %7 : tensor<i32>
  // CHECK-NEXT:     "dataflow.return"(%13) : (tensor<i32>) -> ()
  // CHECK-NEXT:   }) : () -> tensor<i32>
  // CHECK-NEXT:   %9:2 = "dataflow.demux"(%6, %3) : (tensor<i1>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  // CHECK-NEXT:   "dataflow.void"(%9#1) : (tensor<i32>) -> ()
  // CHECK-NEXT:   %10:2 = "dataflow.demux"(%6, %8) : (tensor<i1>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  // CHECK-NEXT:   "dataflow.void"(%10#0) : (tensor<i32>) -> ()
  // CHECK-NEXT:   %11 = "dataflow.mux"(%6, %9#0, %10#1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT:   %12:2 = "dataflow.demux"(%6, %11) : (tensor<i1>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  // CHECK-NEXT:   "dataflow.return"(%6, %12#1, %12#0) : (tensor<i1>, tensor<i32>, tensor<i32>) -> ()
  // CHECK-NEXT: }) : (tensor<i32>) -> tensor<i32>
  %0 = "xla_hlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<i32>):
    %1 = xla_hlo.constant dense<10> : tensor<i32>
    %2 = "xla_hlo.compare"(%arg1, %1) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "xla_hlo.return"(%2) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg1: tensor<i32>):
    %1 = xla_hlo.constant dense<1> : tensor<i32>
    %2 = "xla_hlo.add"(%arg1, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%2) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> tensor<i32>
  return
}

//===----------------------------------------------------------------------===//
// Fork op insertion.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: needsForkOp
func @needsForkOp(%arg0: tensor<i32>) {
  // CHECK-NEXT: %0 = "dataflow.unit_rate"() ( {
  // CHECK-NEXT:   %3 = xla_hlo.constant dense<1> : tensor<i32>
  // CHECK-NEXT:   "dataflow.return"(%3) : (tensor<i32>) -> ()
  // CHECK-NEXT: }) : () -> tensor<i32>
  %0 = xla_hlo.constant dense<1> : tensor<i32>
  // CHECK-NEXT: %1 = "dataflow.fork"(%0) : (tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: %2 = "dataflow.unit_rate"() ( {
  // CHECK-NEXT:   %3 = xla_hlo.add %1, %1 : tensor<i32>
  // CHECK-NEXT:   "dataflow.return"(%3) : (tensor<i32>) -> ()
  // CHECK-NEXT: }) : () -> tensor<i32>
  %1 = "xla_hlo.add"(%0, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return
}
