#include "tensorflow/compiler/mlir/rtl/ir/rtl_ops.h"

namespace mlir {

#include "tensorflow/compiler/mlir/rtl/ir/rtl_structs.cc.inc"

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

unsigned ModuleOp::getNumFuncArguments() {
  getType().getInputs().size();
}

unsigned ModuleOp::getNumFuncResults() {
  getType().getInputs().size();
}

LogicalResult ModuleOp::verifyType() {
  auto type = getType();
  if (!type.isa<FunctionType>()) {
    return emitOpError("requires type attribute of function type");
  }

  return success();
}

} // end namespace rtl
} // end namespace mlir
