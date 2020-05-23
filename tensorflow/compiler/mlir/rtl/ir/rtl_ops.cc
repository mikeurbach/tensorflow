#include "tensorflow/compiler/mlir/rtl/ir/rtl_ops.h"

namespace mlir {

#include "tensorflow/compiler/mlir/rtl/ir/rtl_structs.cc.inc"

namespace rtl {

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

static void print(OpAsmPrinter &p, ModuleOp op) {
}
static void print(OpAsmPrinter &p, ReturnOp op) {
}
static void print(OpAsmPrinter &p, WireOp op) {
}
static void print(OpAsmPrinter &p, InstanceOp op) {
}
static void print(OpAsmPrinter &p, AddOp op) {
}
static void print(OpAsmPrinter &p, AndOp op) {
}
static void print(OpAsmPrinter &p, ConstantOp op) {
}
static void print(OpAsmPrinter &p, DivOp op) {
}
static void print(OpAsmPrinter &p, MulOp op) {
}
static void print(OpAsmPrinter &p, NegateOp op) {
}
static void print(OpAsmPrinter &p, NotOp op) {
}
static void print(OpAsmPrinter &p, OrOp op) {
}
static void print(OpAsmPrinter &p, PowOp op) {
}
static void print(OpAsmPrinter &p, RemOp op) {
}
static void print(OpAsmPrinter &p, ShiftLeftOp op) {
}
static void print(OpAsmPrinter &p, ShiftRightOp op) {
}
static void print(OpAsmPrinter &p, SubOp op) {
}
static void print(OpAsmPrinter &p, XorOp op) {
}

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/rtl/ir/rtl_ops.cc.inc"

} // end namespace rtl
} // end namespace mlir
