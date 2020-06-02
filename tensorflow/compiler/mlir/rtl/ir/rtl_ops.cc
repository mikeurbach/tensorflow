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
  getType().getResults().size();
}

LogicalResult ModuleOp::verifyType() {
  auto type = getType();
  if (!type.isa<FunctionType>()) {
    return emitOpError("requires type attribute of function type");
  }

  return success();
}

static void printType(OpAsmPrinter &p, Type type) {
  assert(type.isa<RankedTensorType>() && "only RankedTensorType is currently supported");
  unsigned bitWidth = type.cast<RankedTensorType>().getElementTypeBitWidth();

  // wires are the implicit default, only print a type if larger than a wire
  if(bitWidth > 1) {
    p << " signed [";
    p << bitWidth - 1;
    p << ":0]";
  }
}

static void printWireDeclaration(OpAsmPrinter &p, Type type) {
  p << "wire";
  printType(p, type);
}

static void printValueName(OpAsmPrinter &p, Value value) {
  // print the value into a separate string
  std::string s;
  llvm::raw_string_ostream os(s);
  p.printOperand(value, os);

  // drop the leading "%"
  StringRef name = StringRef(os.str()).drop_front(1);

  if(value.getKind() == Value::Kind::BlockArgument) {
    // value is an input to a ModuleOp, use its name directly
    p << name;
  } else {
    // value is an SSA result, prepend "tmp" to its SSA ID
    p << "tmp" << name;
  }
}

static void printUnaryOp(OpAsmPrinter &p, Value result, Value arg, StringRef op) {
  printWireDeclaration(p, result.getType());
  p << " ";
  printValueName(p, result);
  p << " = ";
  p << op;
  printValueName(p, arg);
  p << ";";
}

static void printBinaryOp(OpAsmPrinter &p, Value result, Value left, Value right, StringRef op) {
  printWireDeclaration(p, result.getType());
  p << " ";
  printValueName(p, result);
  p << " = ";
  printValueName(p, left);
  p << " " << op << " ";
  printValueName(p, right);
  p << ";";
}

static void printPorts(OpAsmPrinter &p, FunctionType type) {
  unsigned numInputs = type.getNumInputs();
  unsigned numOutputs = type.getNumResults();

  p << " (\n";

  for(unsigned i = 0; i < numInputs; ++i) {
    Type inputType = type.getInput(i);
    p << "    input";
    printType(p, inputType);
    p << " arg" << std::to_string(i);
    if(numOutputs > 0 || i < numInputs - 1) {
      p << ",";
    }
    p << "\n";
  }

  for(unsigned i = 0; i < numOutputs; ++i) {
    Type outputType = type.getResult(i);
    p << "    output";
    printType(p, outputType);
    p << " ret" << std::to_string(i);
    if(i < numOutputs - 1) {
      p << ",";
    }
    p << "\n";
  }

  p << "  );";
}

static void print(OpAsmPrinter &p, ModuleOp op) {
  p << "module ";
  p << SymbolTable::getSymbolName(op);
  printPorts(p, op.type());
  p << "\n";
  for(auto &op : op.getBody().front()) {
    p << "    ";
    auto abstractOp = op.getAbstractOperation();
    abstractOp->printAssembly(&op, p);
    p << "\n";
  }
  p << "  endmodule\n";
}

static void print(OpAsmPrinter &p, InputOp op) {
  unsigned numArgs = op.getNumOperands();
  for(unsigned i = 0; i < numArgs; ++i) {
    if(i > 0) {
      p << "    ";
    }
    p << "assign ";
    p << cast<WireOp>(op.getOperand(i).getDefiningOp()).name();
    p << " = ";
    p << "arg" << i << ";";
    if(i < numArgs - 1) {
      p << "\n";
    }
  }
}

static void print(OpAsmPrinter &p, ReturnOp op) {
  unsigned numOperands = op.getNumOperands();
  for(unsigned i = 0; i < numOperands; ++i) {
    if(i > 0) {
      p << "    ";
    }
    p << "assign ";
    p << "ret" << i;
    p << " = ";
    ModuleOp parent = op.getParentOfType<ModuleOp>();
    if(SymbolTable::getSymbolName(parent).compare("main") == 0) {
      // for main function, we're returning specific named wires
      p << cast<WireOp>(op.getOperand(i).getDefiningOp()).name();
    } else {
      // for every other function, we're returning the generated wire name
      printValueName(p, op.getOperand(i));
    }
    p << ";";
    if(i < numOperands - 1) {
      p << "\n";
    }
  }
}

static void print(OpAsmPrinter &p, WireOp op) {
  printWireDeclaration(p, op.value().getType());
  p << " " << op.name();
  p << ";";
}

static unsigned instanceCounter = 0;
static void print(OpAsmPrinter &p, InstanceOp op) {
  SymbolRefAttr instantiation = op.instantiation();
  StringRef moduleName = instantiation.getRootReference();
  ArrayRef<FlatSymbolRefAttr> ports = op.instantiation().getNestedReferences();

  mlir::ModuleOp topModule = op.getParentOfType<mlir::ModuleOp>();
  ModuleOp module = cast<ModuleOp>(topModule.lookupSymbol(moduleName));
  unsigned numArgs = module.getNumFuncArguments();
  unsigned numResults = module.getNumFuncResults();

  p << moduleName;
  p << " instance_";
  p << ++instanceCounter;
  p << " (\n";
  for(unsigned i = 0; i < numArgs; ++i) {
    p << "      .arg" << i;
    p << "(" << ports[i].getValue() << ")";
    if(numResults > 0 || i < numArgs - 1) {
      p << ",";
    }
    p << "\n";
  }
  for(unsigned i = 0; i < numResults; ++i) {
    p << "      .ret" << i;
    p << "(" << ports[numArgs + i].getValue() << ")";
    if(i < numResults - 1) {
      p << ",";
    }
    p << "\n";
  }
  p << "    );";
}

static void print(OpAsmPrinter &p, ConstantOp op) {
  Value result = op.result();
  printWireDeclaration(p, result.getType());
  p << " ";
  printValueName(p, result);
  p << " = ";
  APInt constant = *op.value().begin();
  unsigned bitWidth = constant.getBitWidth();
  if(bitWidth == 1) {
    p << "1'b" << (constant.getBoolValue() ? "1" : "0");
  } else {
    p << bitWidth << "'d" << constant;
  }
  p << ";";
}

static void print(OpAsmPrinter &p, AddOp op) {
  printBinaryOp(p, op.result(), op.left(), op.right(), "+");
}
static void print(OpAsmPrinter &p, AndOp op) {
  printBinaryOp(p, op.result(), op.left(), op.right(), "&");
}
static void print(OpAsmPrinter &p, DivOp op) {
  printBinaryOp(p, op.result(), op.left(), op.right(), "/");
}
static void print(OpAsmPrinter &p, MulOp op) {
  printBinaryOp(p, op.result(), op.left(), op.right(), "*");
}
static void print(OpAsmPrinter &p, NegateOp op) {
  printUnaryOp(p, op.result(), op.arg(), "-");
}
static void print(OpAsmPrinter &p, NotOp op) {
  printUnaryOp(p, op.result(), op.arg(), "!");
}
static void print(OpAsmPrinter &p, OrOp op) {
  printBinaryOp(p, op.result(), op.left(), op.right(), "|");
}
static void print(OpAsmPrinter &p, PowOp op) {
  printBinaryOp(p, op.result(), op.left(), op.right(), "**");
}
static void print(OpAsmPrinter &p, RemOp op) {
  printBinaryOp(p, op.result(), op.left(), op.right(), "%");
}
static void print(OpAsmPrinter &p, ShiftLeftOp op) {
  printBinaryOp(p, op.result(), op.left(), op.right(), "<<");
}
static void print(OpAsmPrinter &p, ShiftRightOp op) {
  printBinaryOp(p, op.result(), op.left(), op.right(), ">>");
}
static void print(OpAsmPrinter &p, SubOp op) {
  printBinaryOp(p, op.result(), op.left(), op.right(), "-");
}
static void print(OpAsmPrinter &p, XorOp op) {
  printBinaryOp(p, op.result(), op.left(), op.right(), "^");
}

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/rtl/ir/rtl_ops.cc.inc"

} // end namespace rtl
} // end namespace mlir
