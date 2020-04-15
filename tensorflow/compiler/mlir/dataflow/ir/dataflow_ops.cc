#include "tensorflow/compiler/mlir/dataflow/ir/dataflow_ops.h"

namespace mlir {
namespace dataflow {

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/dataflow/ir/dataflow_ops.cc.inc"

DataflowDialect::DataflowDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/dataflow/ir/dataflow_ops.cc.inc"
    >();
}

} // end namespace dataflow
} // end namespace mlir
