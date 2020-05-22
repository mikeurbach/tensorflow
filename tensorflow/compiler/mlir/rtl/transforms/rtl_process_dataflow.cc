#include <numeric>
#include <unordered_map>
#include "llvm/ADT/SmallString.h"  // TF:llvm-project
#include "llvm/ADT/SmallVector.h"  // TF:llvm-project
#include "llvm/ADT/StringSet.h"  // TF:llvm-project
#include "llvm/ADT/Twine.h"  // TF:llvm-project
#include "llvm/Support/Debug.h"  // TF:llvm-project
#include "llvm/Support/ScopedPrinter.h"  // TF:llvm-project
#include "llvm/Support/raw_ostream.h"  // TF:llvm-project
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
 public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    OpBuilder builder = OpBuilder(module.getBody()->getTerminator());

    WiringTable wiringTable;

    module.walk(
        [&](FuncOp function) {
          function.walk(
              [&](dataflow::UnitRateOp op) {
                liftUnitRateOp(op, builder, &wiringTable);
              });

          connectMainPorts(function, builder, &wiringTable);

          buildMainBody(function, builder, wiringTable);
        });
  }
 private:
  class WiringTable {
   public:
    enum class PortType { INPUT, OUTPUT };

    WiringTable() {
      this->wireLookup = wireLookupType();
      this->wireTypes = llvm::StringMap<Type>();
    }

    void addWire(Type wireType,
        std::string sourceSymbol, PortType sourceType, unsigned sourceNumber,
        std::string sinkSymbol, PortType sinkType, unsigned sinkNumber) {
      // generate a wire name
      std::string wireName = "";
      wireName += sourceSymbol + "_";
      wireName += portTypeName(sourceType) + "_";
      wireName += std::to_string(sourceNumber) + "_to_";
      wireName += sinkSymbol + "_";
      wireName += portTypeName(sinkType) + "_";
      wireName += std::to_string(sinkNumber);

      // track for source
      wireLookup[sourceSymbol][sourceType][sourceNumber] = wireName;

      // track for sink
      wireLookup[sinkSymbol][sinkType][sinkNumber] = wireName;

      // track type
      wireTypes[wireName] = wireType;
    }

    llvm::StringSet<> getWires() {
      llvm::StringSet<> wires;

      for(auto funcIt = wireLookup.begin();
          funcIt != wireLookup.end(); ++funcIt) {
        for(auto portTypeIt = funcIt->second.begin();
            portTypeIt != funcIt->second.end(); ++portTypeIt) {
          for(auto portIt = portTypeIt->second.begin();
              portIt != portTypeIt->second.end(); ++portIt) {
            wires.insert(portIt->second);
          }
        }
      }

      return wires;
    }

    Type getWireType(std::string wireName) {
      return wireTypes[wireName];
    }

    llvm::StringSet<> getFuncs() {
      llvm::StringSet<> funcs;

      for(auto funcIt = wireLookup.begin();
          funcIt != wireLookup.end(); ++funcIt) {
        funcs.insert(funcIt->first);
      }

      return funcs;
    }

    std::unordered_map<unsigned, std::string> lookupFuncPorts(
        std::string funcName, PortType portType) {
      return wireLookup.at(funcName).at(portType);
    }

    std::string print() {
      std::string out = "WiringTable:\n";
      for(auto funcIt = wireLookup.begin(); funcIt != wireLookup.end(); ++funcIt) {
        out += funcIt->first + "\n";
        for(auto portTypeIt = funcIt->second.begin();
            portTypeIt != funcIt->second.end(); ++portTypeIt) {
          out += "  " + portTypeName(portTypeIt->first) + "\n";
          for(auto portIt = portTypeIt->second.begin();
              portIt != portTypeIt->second.end(); ++portIt) {
            out += "    " + std::to_string(portIt->first) + ": " + portIt->second + "\n";
          }
        }
        out += "\n";
      }
      return out;
    }
   private:
    // func symbol + port type + port number -> wire name
    using wireLookupType =
      std::unordered_map<
        std::string,
        std::unordered_map<
          PortType,
          std::unordered_map<
            unsigned, std::string>>>;

    wireLookupType wireLookup;

    llvm::StringMap<Type> wireTypes;

    std::string portTypeName(PortType portType) {
      return portType == PortType::INPUT ? "in" : "out";
    }

    llvm::ScopedPrinter logger{llvm::dbgs()};
  };

  void buildMainBody(FuncOp function, OpBuilder builder, WiringTable wiringTable) {
    // create top-level main function
    FunctionType liftedType = liftedFunctionType(
        builder, function.getType().getInputs(), function.getType().getResults());

    std::string liftedName = MAIN_FUNCTION_NAME;

    ArrayRef<NamedAttribute> liftedAttrs = {};

    FuncOp main = builder.create<FuncOp>(
        function.getLoc(), liftedName, liftedType, liftedAttrs);

    // add a body to the top-level function
    Block *entry = main.addEntryBlock();
    OpBuilder::InsertionGuard entryInsertionGuard(builder);
    builder.setInsertionPointToStart(entry);

    // insert every wire into the top-level function
    llvm::StringSet<> wireSet = wiringTable.getWires();
    std::unordered_map<std::string, Value> wireMap;
    for(auto it = wireSet.begin(); it != wireSet.end(); ++it) {
      StringRef wireName = it->getKey();
      Type wireType = wiringTable.getWireType(wireName.str());
      WireOp wireOp = builder.create<WireOp>(main.getLoc(), wireType, wireName);
      // keep track of the result by name for return values
      wireMap[wireName.str()] = wireOp.getResult();
    }

    // insert every instance into the top-level function
    llvm::StringSet<> funcSet = wiringTable.getFuncs();
    for(auto it = funcSet.begin(); it != funcSet.end(); ++it) {
      StringRef funcName = it->getKey();
      if(funcName.compare(MAIN_FUNCTION_NAME) == 0) {
        continue;
      }

      auto funcInputs = wiringTable.lookupFuncPorts(
          funcName.str(), WiringTable::PortType::INPUT);
      auto funcOutputs = wiringTable.lookupFuncPorts(
          funcName.str(), WiringTable::PortType::OUTPUT);

      std::vector<FlatSymbolRefAttr> funcWires(funcInputs.size() + funcOutputs.size());
      for(auto fi = funcInputs.begin(); fi != funcInputs.end(); ++fi) {
        funcWires[fi->first] = builder.getSymbolRefAttr(fi->second);
      }
      for(auto fi = funcOutputs.begin(); fi != funcOutputs.end(); ++fi) {
        funcWires[funcInputs.size() + fi->first] = builder.getSymbolRefAttr(fi->second);
      }

      SymbolRefAttr instantiation = builder.getSymbolRefAttr(funcName, funcWires);
      builder.create<InstanceOp>(main.getLoc(), instantiation);
    }

    // insert a return with every output wire in the top-level function
    auto mainOutputs = wiringTable.lookupFuncPorts(
        MAIN_FUNCTION_NAME, WiringTable::PortType::OUTPUT);

    std::vector<Value> outputWires(mainOutputs.size());
    for(auto it = mainOutputs.begin(); it != mainOutputs.end(); ++it) {
      outputWires[it->first] = wireMap[it->second];
    }

    builder.create<ReturnOp>(main.getLoc(), outputWires);
  }

  void connectMainPorts(FuncOp function, OpBuilder builder, WiringTable *wiringTable) {
    Block &body = function.front();

    // find input uses
    for(unsigned i = 0; i < body.getNumArguments(); ++i) {
      // get the single use of this result
      BlockArgument arg = body.getArgument(i);
      OpOperand &use = *arg.use_begin();
      Operation &useOp = *use.getOwner();

      // connect input use ready signals to main function's ready outputs
      RankedTensorType bitType = RankedTensorType::get({}, builder.getI1Type());
      wiringTable->addWire(
          bitType,
          liftedFunctionName(useOp), WiringTable::PortType::OUTPUT, use.getOperandNumber(),
          MAIN_FUNCTION_NAME, WiringTable::PortType::OUTPUT, i);
    }

    // find return operation
    Operation &ret = body.back();
    unsigned offset = function.getType().getNumInputs();

    // connect returned result(s) and valid signals to main function's data and valid outputs
    for(unsigned i = 0; i < ret.getNumOperands(); ++i) {
      trackValueSource(
          ret.getOperand(i),
          MAIN_FUNCTION_NAME, WiringTable::PortType::OUTPUT, offset + i,
          wiringTable, builder);
    }
  }

  void liftUnitRateOp(dataflow::UnitRateOp op, OpBuilder builder, WiringTable *wiringTable) {
    Operation &operation = *op.getOperation();

    FunctionType liftedType = liftedFunctionType(
        builder, operation.getOperandTypes(), operation.getResultTypes());

    std::string liftedName = liftedFunctionName(operation);

    ArrayRef<NamedAttribute> liftedAttrs = {};

    FuncOp lifted = builder.create<FuncOp>(
        op.getLoc(), liftedName, liftedType, liftedAttrs);

    configureLiftedFunction(op, lifted, builder, wiringTable);
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

  void configureLiftedFunction(dataflow::UnitRateOp &baseOp, FuncOp lifted, OpBuilder builder,
                               WiringTable *wiringTable) {
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
      Value operand = op.getOperand(i);
      trackValueSource(
          operand,
          SymbolTable::getSymbolName(lifted).str(), WiringTable::PortType::INPUT, dataIndex,
          wiringTable, builder);
    }

    // wire up ready signals
    for(unsigned i = 0; i < op.getNumResults(); ++i) {
      // get the single use of this result
      OpResult result = op.getResult(i);
      OpOperand &use = *result.use_begin();
      Operation &useOp = *use.getOwner();

      // find the source for every result ready signal
      std::string sourceFunc;
      WiringTable::PortType sourceKind;
      unsigned sourceIndex;
      if(useOp.getName() == OperationName("std.return", op.getContext())) {
        // source is an output to the top level function
        sourceFunc = MAIN_FUNCTION_NAME;
        sourceKind = WiringTable::PortType::INPUT;
        FuncOp parent = static_cast<FuncOp>(useOp.getParentOp());
        FunctionType parentType = parent.getType();
        sourceIndex = ((parentType.getNumInputs() * 2) + 1) - parentType.getNumResults() + i;
      } else {
        // source is the result of an operation
        sourceFunc = liftedFunctionName(useOp);
        sourceKind = WiringTable::PortType::OUTPUT;
        sourceIndex = use.getOperandNumber();
      }

      // save the ready source
      unsigned readySourceNumber = entry->getNumArguments() - op.getNumResults() + i;
      wiringTable->addWire(
          bitType,
          sourceFunc, sourceKind, sourceIndex,
          SymbolTable::getSymbolName(lifted).str(), WiringTable::PortType::INPUT, readySourceNumber);
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

  void trackValueSource(
      Value value,
      std::string destFunc, WiringTable::PortType destKind, unsigned destIndex,
      WiringTable *wiringTable, OpBuilder builder) {
    std::string sourceFunc;
    WiringTable::PortType sourceKind;
    unsigned sourceDataIndex;
    unsigned sourceValidIndex;
    switch(value.getKind()) {
      case mlir::Value::Kind::OpResult0:
      case mlir::Value::Kind::OpResult1:
      case mlir::Value::Kind::TrailingOpResult: {
        // source is the result of an operation
        mlir::OpResult opResult = static_cast<OpResult&>(value);
        Operation *sourceOp = opResult.getDefiningOp();
        sourceFunc = liftedFunctionName(*sourceOp);
        sourceKind = WiringTable::PortType::OUTPUT;
        // every source operand will have an output for ready signal, so index past those
        unsigned offset = sourceOp->getNumOperands();
        sourceDataIndex = offset + opResult.getResultNumber();
        sourceValidIndex = offset + opResult.getResultNumber() + 1;
        break;
      }
      case mlir::Value::Kind::BlockArgument: {
        // source is an input to the top level function
        mlir::BlockArgument blockArgument = static_cast<BlockArgument&>(value);
        Operation *sourceOp = blockArgument.getOwner()->getParentOp();
        sourceFunc = MAIN_FUNCTION_NAME;
        sourceKind = WiringTable::PortType::INPUT;
        unsigned idx = blockArgument.getArgNumber() * 2;
        sourceDataIndex = idx;
        sourceValidIndex = idx + 1;
        break;
      }
    }

    // save the data source
    wiringTable->addWire(
        value.getType(),
        sourceFunc, sourceKind, sourceDataIndex,
        destFunc, destKind, destIndex);

    // save the valid source
    RankedTensorType bitType = RankedTensorType::get({}, builder.getI1Type());
    wiringTable->addWire(
        bitType,
        sourceFunc, sourceKind, sourceValidIndex,
        destFunc, destKind, destIndex + 1);
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
