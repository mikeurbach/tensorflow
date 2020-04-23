#include "tensorflow/compiler/mlir/rtl/ir/rtl_ops.h"

namespace mlir {
namespace rtl {

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/rtl/ir/rtl_ops.cc.inc"

RTLDialect::RTLDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/rtl/ir/rtl_ops.cc.inc"
    >();
}

} // end namespace rtl
} // end namespace mlir
