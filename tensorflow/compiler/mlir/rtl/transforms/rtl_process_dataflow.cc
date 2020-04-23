#include "llvm/ADT/SmallVector.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassManager.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/rtl/ir/rtl_ops.h"
#include "tensorflow/compiler/mlir/dataflow/ir/dataflow_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace rtl {

#include "tensorflow/compiler/mlir/rtl/transforms/rtl_process_dataflow.cc.inc"

class LegalizeDataflow : public PassWrapper<LegalizeDataflow, FunctionPass> {
  void runOnFunction() override {
    Operation *op = getFunction();

    MLIRContext *context = op->getContext();

    ConversionTarget target(*context);

    target.addLegalDialect<RTLDialect>();

    OwningRewritePatternList patterns;

    populateWithGenerated(context, &patterns);

    LogicalResult result = applyPartialConversion(op, target, patterns);

    if(failed(result)) {
      signalPassFailure();
    }
  }
};

static void pipelineBuilder(OpPassManager &passManager) {
  passManager.addPass(std::make_unique<LegalizeDataflow>());
}

static PassPipelineRegistration<> pipeline(
    "rtl-process-dataflow",
    "Process modules in the Dataflow dialect to generate RTL modules",
    pipelineBuilder);

} // end namespace rtl
} // end namespace mlir
