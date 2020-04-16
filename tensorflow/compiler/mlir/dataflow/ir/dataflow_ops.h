#ifndef TENSORFLOW_COMPILER_MLIR_DATAFLOW_IR_DATAFLOW_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_DATAFLOW_IR_DATAFLOW_OPS_H_

#include "mlir/IR/Dialect.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Interfaces/SideEffects.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace dataflow {

class DataflowDialect : public Dialect {
public:
  explicit DataflowDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "dataflow"; }
};

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/dataflow/ir/dataflow_ops.h.inc"

} // end namespace dataflow
} // end namespace mlir

#endif
