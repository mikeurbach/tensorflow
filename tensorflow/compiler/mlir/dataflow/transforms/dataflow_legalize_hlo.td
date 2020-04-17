include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/dataflow/ir/dataflow_ops.td"
include "tensorflow/compiler/mlir/xla/ir/hlo_ops.td"

def LegalizeReturn : Pat<(HLO_ReturnOp $results), (Dataflow_ReturnOp $results)>;
