#include <numeric>
#include "llvm/ADT/SmallString.h"  // TF:llvm-project
#include "llvm/ADT/SmallVector.h"  // TF:llvm-project
#include "llvm/ADT/Twine.h"  // TF:llvm-project
#include "llvm/Support/Debug.h"  // TF:llvm-project
#include "llvm/Support/ScopedPrinter.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/SymbolTable.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassManager.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/rtl/ir/rtl_ops.h"
#include "tensorflow/compiler/mlir/dataflow/ir/dataflow_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

#define DEBUG_TYPE "rtl-process-dataflow"

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

class LiftOpsToFunctions : public PassWrapper<LiftOpsToFunctions, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk(
        [&](FuncOp function) {
          function.walk(
              [&](dataflow::UnitRateOp op) {
                liftUnitRateOp(op, module);
              });
        });
  }

 private:
  void liftUnitRateOp(dataflow::UnitRateOp op, ModuleOp parentModule) {
    LLVM_DEBUG(logger.startLine() << "lifting " << op.getOperationName() << "\n");

    Operation &inner = op.body().front().front();

    FunctionType liftedType = liftedFunctionType(inner);

    std::string liftedName = liftedFunctionName(*op.getOperation());

    FuncOp lifted = FuncOp::create(op.getLoc(), liftedName, liftedType);

    configureLiftedFunction(inner, lifted);

    parentModule.push_back(lifted.getOperation());
  }

  FunctionType liftedFunctionType(Operation &op) {
    Builder builder(op.getContext());

    RankedTensorType bitType = RankedTensorType::get({}, builder.getI1Type());

    SmallVector<Type, 2> inputTypes;
    SmallVector<Type, 2> outputTypes;

    for(auto operand : op.getOperands()) {
      inputTypes.push_back(operand.getType()); // input for the operand data
      inputTypes.push_back(bitType); // input for the operand valid
      outputTypes.push_back(bitType); // output for the operand ready
    }

    for(auto result : op.getResults()) {
      outputTypes.push_back(result.getType()); // output for the result data
      outputTypes.push_back(bitType); // output for the result valid
      inputTypes.push_back(bitType); // input for the result ready
    }

    return builder.getFunctionType(inputTypes, outputTypes);
  }

  std::string liftedFunctionName(Operation &op) {
    std::string name = "";
    if(op.hasTrait<mlir::OpTrait::Symbol>()){
      name += SymbolTable::getSymbolName(&op);
    } else {
      name += op.getName().getStringRef();
      name += "_";
      name += std::to_string((long) ((const void *) &op));
    }
    return name;
  }

  void configureLiftedFunction(Operation &op, FuncOp lifted) {
    Block *entry = lifted.addEntryBlock();

    OpBuilder builder(entry->getParent());

    RankedTensorType bitType = RankedTensorType::get({}, builder.getI1Type());

    // wire up data signals to wrapped operation
    Operation *operation = builder.clone(op);
    for(unsigned i = 0; i < op.getNumOperands(); ++i) {
      // index to the data signal in the arguments
      unsigned index = i * 2;
      operation->setOperand(i, entry->getArgument(index));

      // find the source function, kind, and index
      FlatSymbolRefAttr sourceFunc;
      StringAttr sourceKind;
      IntegerAttr sourceDataIndex;
      IntegerAttr sourceValidIndex;
      Value operand = op.getOperand(i);
      switch(operand.getKind()) {
        case mlir::Value::Kind::OpResult0:
        case mlir::Value::Kind::OpResult1:
        case mlir::Value::Kind::TrailingOpResult: {
          // source is the result of an operation
          mlir::OpResult opResult = static_cast<OpResult&>(operand);
          Operation *sourceOp = opResult.getDefiningOp();
          sourceFunc = builder.getSymbolRefAttr(liftedFunctionName(*sourceOp));
          sourceKind = builder.getStringAttr("RESULT");
          sourceDataIndex = builder.getI32IntegerAttr(opResult.getResultNumber());
          sourceValidIndex = builder.getI32IntegerAttr(opResult.getResultNumber() + 1);
          break;
        }
        case mlir::Value::Kind::BlockArgument: {
          // source is the argument to the top level function
          mlir::BlockArgument blockArgument = static_cast<BlockArgument&>(operand);
          Operation *sourceOp = blockArgument.getOwner()->getParentOp();
          sourceFunc = builder.getSymbolRefAttr(liftedFunctionName(*sourceOp));
          sourceKind = builder.getStringAttr("ARGUMENT");
          sourceDataIndex = builder.getI32IntegerAttr(blockArgument.getArgNumber());
          sourceValidIndex = builder.getI32IntegerAttr(blockArgument.getArgNumber() + 1);
          break;
        }
      }

      // save the data source
      Source dataSource = Source::get(sourceFunc, sourceKind, sourceDataIndex, op.getContext());
      lifted.setArgAttr(index, "rtl.source", dataSource);

      // save the valid source
      Source validSource = Source::get(sourceFunc, sourceKind, sourceValidIndex, op.getContext());
      lifted.setArgAttr(index + 1, "rtl.source", validSource);
    }

    // compute valid signal for results
    Value valid;
    if(op.getNumOperands() == 0) {
      // with no inputs, always valid
      ConstantOp const1 = builder.create<ConstantOp>(
          op.getLoc(), bitType, DenseElementsAttr::get(bitType, APInt(1, 1)));
      valid = const1.getResult();
    } else {
      // with inputs, valid when they all are
      SmallVector<Value, 2> operandValids;

      // index to the valid signal in the arguments
      for(unsigned i = 0; i < op.getNumOperands(); ++i) {
        operandValids.push_back(entry->getArgument((i * 2) + 1));
      }

      // combine valid signals in a balanced tree of two-input adders
      // returns the obvious for one or two operands, the most common,
      // and the larger cases could be optimized later
      valid = std::accumulate(operandValids.begin() + 1, operandValids.end(), operandValids[0],
                              [&builder, &op, &bitType](Value operandValid1, Value operandValid2) {
                                AndOp andValid = builder.create<AndOp>(
                                    op.getLoc(), bitType, operandValid1, operandValid2);
                                return andValid.getResult();
                              });
    }

    SmallVector<Value, 3> results;

    // can skip ready signal for constant ops
    if(op.getNumOperands() > 0) {
      // compute ready signal for operands
      SmallVector<Value, 1> resultsReady;
      for(unsigned i = entry->getNumArguments() - 1, j = 0; j < op.getNumResults(); --i, ++j) {
        // index to the ready signals at the end of the arguments
        resultsReady.push_back(entry->getArgument(i));
      }
      resultsReady.push_back(valid);
      AndOp validAndReady = builder.create<AndOp>(
          op.getLoc(), bitType, resultsReady, SmallVector<NamedAttribute, 0>());

      // return the ready bits for the inputs
      for(unsigned i = 0; i < op.getNumOperands(); ++i) {
        results.push_back(validAndReady);
      }
    }

    // return the data and valid bits
    results.push_back(operation->getResult(0));
    results.push_back(valid);

    builder.create<ReturnOp>(op.getLoc(), results);
  }

  llvm::ScopedPrinter logger{llvm::dbgs()};
};

static void pipelineBuilder(OpPassManager &passManager) {
  passManager.addPass(std::make_unique<LegalizeDataflow>());
  passManager.addPass(std::make_unique<LiftOpsToFunctions>());
}

static PassPipelineRegistration<> pipeline(
    "rtl-process-dataflow",
    "Process modules in the Dataflow dialect to generate RTL modules",
    pipelineBuilder);

} // end namespace rtl
} // end namespace mlir
