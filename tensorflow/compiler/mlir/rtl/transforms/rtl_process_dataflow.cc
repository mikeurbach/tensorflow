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

    OpBuilder builder = OpBuilder(module.getBody()->getTerminator());

    module.walk(
        [&](FuncOp function) {
          function.walk(
              [&](dataflow::UnitRateOp op) {
                liftUnitRateOp(op, builder);
              });

          liftMainFunc(function, builder);
        });
  }

 private:
  void liftMainFunc(FuncOp function, OpBuilder builder) {
    Block &body = function.getCallableRegion()->front();

    FunctionType liftedType = liftedFunctionType(
        builder, function.getType().getInputs(), function.getType().getResults());

    std::string liftedName = MAIN_FUNCTION_NAME;

    ArrayRef<NamedAttribute> liftedAttrs = {};

    FuncOp main = builder.create<FuncOp>(
        function.getLoc(), liftedName, liftedType, liftedAttrs);
  }

  void liftUnitRateOp(dataflow::UnitRateOp op, OpBuilder builder) {
    LLVM_DEBUG(logger.startLine() << "lifting " << op.getOperationName() << "\n");

    Operation &operation = *op.getOperation();

    FunctionType liftedType = liftedFunctionType(
        builder, operation.getOperandTypes(), operation.getResultTypes());

    std::string liftedName = liftedFunctionName(operation);

    ArrayRef<NamedAttribute> liftedAttrs = {};

    FuncOp lifted = builder.create<FuncOp>(
        op.getLoc(), liftedName, liftedType, liftedAttrs);

    configureLiftedFunction(op, lifted, builder);
  }

  FunctionType liftedFunctionType(
      Builder builder, TypeRange operandTypes, TypeRange resultTypes) {
    RankedTensorType bitType = RankedTensorType::get({}, builder.getI1Type());

    SmallVector<Type, 2> inputTypes;
    SmallVector<Type, 2> outputTypes;

    for(Type operandType : operandTypes) {
      inputTypes.push_back(operandType); // input for the operand data
      inputTypes.push_back(bitType); // input for the operand valid
      outputTypes.push_back(bitType); // output for the operand ready
    }

    for(Type resultType : resultTypes) {
      outputTypes.push_back(resultType); // output for the result data
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

  void configureLiftedFunction(dataflow::UnitRateOp &baseOp, FuncOp lifted, OpBuilder builder) {
    Operation &op = *baseOp.getOperation();

    Operation &inner = baseOp.body().front().front();

    Block *entry = lifted.addEntryBlock();

    OpBuilder::InsertionGuard entryInsertionGuard(builder);
    builder.setInsertionPointToStart(entry);

    RankedTensorType bitType = RankedTensorType::get({}, builder.getI1Type());

    // wire up data and valid signals to wrapped operation
    Operation *operation = builder.clone(inner);
    for(unsigned i = 0; i < op.getNumOperands(); ++i) {
      // index to the data signal in the arguments
      unsigned dataIndex = i * 2;
      operation->setOperand(i, entry->getArgument(dataIndex));

      // find the source for every data signal
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
          // every source operand will have an output for ready signal, so index past those
          unsigned offset = sourceOp->getNumOperands();
          sourceDataIndex = builder.getI32IntegerAttr(offset + opResult.getResultNumber());
          sourceValidIndex = builder.getI32IntegerAttr(offset + opResult.getResultNumber() + 1);
          break;
        }
        case mlir::Value::Kind::BlockArgument: {
          // source is an input to the top level function
          mlir::BlockArgument blockArgument = static_cast<BlockArgument&>(operand);
          Operation *sourceOp = blockArgument.getOwner()->getParentOp();
          sourceFunc = builder.getSymbolRefAttr(MAIN_FUNCTION_NAME);
          sourceKind = builder.getStringAttr("ARGUMENT");
          unsigned idx = blockArgument.getArgNumber() * 2;
          sourceDataIndex = builder.getI32IntegerAttr(idx);
          sourceValidIndex = builder.getI32IntegerAttr(idx + 1);
          break;
        }
      }

      // save the data source
      Source dataSource = Source::get(sourceFunc, sourceKind, sourceDataIndex, op.getContext());
      lifted.setArgAttr(dataIndex, "rtl.source", dataSource);

      // save the valid source
      Source validSource = Source::get(sourceFunc, sourceKind, sourceValidIndex, op.getContext());
      lifted.setArgAttr(dataIndex + 1, "rtl.source", validSource);
    }

    // wire up ready signals
    for(unsigned i = 0; i < op.getNumResults(); ++i) {
      // get the single use of this result
      OpResult result = op.getResult(i);
      OpOperand &use = *result.use_begin();
      Operation &useOp = *use.getOwner();

      // find the source for every result ready signal
      StringAttr sourceKind;
      FlatSymbolRefAttr sourceFunc;
      IntegerAttr sourceIndex;
      if(useOp.getName() == OperationName("std.return", op.getContext())) {
        // source is an output to the top level function
        sourceKind = builder.getStringAttr("ARGUMENT");
        sourceFunc = builder.getSymbolRefAttr(MAIN_FUNCTION_NAME);
        FuncOp parent = static_cast<FuncOp>(useOp.getParentOp());
        FunctionType parentType = parent.getType();
        unsigned idx = ((parentType.getNumInputs() * 2) + 1) - parentType.getNumResults() + i;
        sourceIndex = builder.getI32IntegerAttr(idx);
      } else {
        // source is the result of an operation
        sourceKind = builder.getStringAttr("RESULT");
        sourceFunc = builder.getSymbolRefAttr(liftedFunctionName(useOp));
        sourceIndex = builder.getI32IntegerAttr(use.getOperandNumber());
      }

      // save the ready source
      Source readySource = Source::get(sourceFunc, sourceKind, sourceIndex, op.getContext());
      lifted.setArgAttr(entry->getNumArguments() - op.getNumResults() + i, "rtl.source", readySource);
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
      for(unsigned i = entry->getNumArguments() - op.getNumResults(); i < entry->getNumArguments(); ++i) {
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

  inline static const std::string MAIN_FUNCTION_NAME = "main";
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
