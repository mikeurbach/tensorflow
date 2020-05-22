#ifndef TENSORFLOW_COMPILER_MLIR_RTL_IR_RTL_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_RTL_IR_RTL_OPS_H_

#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Dialect.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Interfaces/SideEffects.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/rtl/ir/rtl_ops.h"

namespace mlir {

#include "tensorflow/compiler/mlir/rtl/ir/rtl_structs.h.inc"

namespace rtl {

class RTLDialect : public Dialect {
public:
  explicit RTLDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "rtl"; }
};

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/rtl/ir/rtl_ops.h.inc"

} // end namespace rtl
} // end namespace mlir

#endif
