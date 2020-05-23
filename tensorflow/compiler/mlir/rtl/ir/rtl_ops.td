#ifndef RTL_OPS
#define RTL_OPS

include "mlir/Interfaces/SideEffects.td"
include "mlir/IR/OpBase.td"

// base

def RTL_Dialect : Dialect {
    let name = "rtl";
    let cppNamespace = "rtl";
}

class RTL_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<RTL_Dialect, mnemonic, traits> {
}

// types

def RTL_ControlType : AnyTypeOf<
    [0DTensorOf<[AnySignlessInteger]>]>;

def RTL_DataType : AnyTypeOf<
    [0DTensorOf<[AnySignlessInteger]>, TupleOf<[0DTensorOf<[AnySignlessInteger]>]>]>;

// unit rate ops

def RTL_ConstantOp : RTL_Op<"constant",
    [ConstantLike, NoSideEffect]> {
  let summary = [{
    Constant op.
  }];

  let description = [{
    An operation that produces a value that is always valid.
  }];

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    RTL_DataType:$results
  );
}

// arithmetic

class RTL_UnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []>
    : RTL_Op<mnemonic, !listconcat(traits, [NoSideEffect, SameOperandsAndResultType])> {
  let arguments = (ins
    RTL_DataType:$arg
  );

  let results = (outs
    RTL_DataType:$result
  );
}

def RTL_NegateOp : RTL_UnaryArithmeticOp<"negate"> {
  let summary = [{
    Negate op.
  }];

  let description = [{
    An operation to represent unary negate.
  }];
}

class RTL_BinaryArithmeticOp<string mnemonic, list<OpTrait> traits = []>
    : RTL_Op<mnemonic, !listconcat(traits, [NoSideEffect, SameOperandsAndResultType])> {
  let arguments = (ins
    RTL_DataType:$left,
    RTL_DataType:$right
  );

  let results = (outs
    RTL_DataType:$result
  );
}

def RTL_AddOp : RTL_BinaryArithmeticOp<"add", [Commutative]> {
  let summary = [{
    Add op.
  }];

  let description = [{
    An operation to represent binary add.
  }];
}

def RTL_DivOp : RTL_BinaryArithmeticOp<"div"> {
  let summary = [{
    Div op.
  }];

  let description = [{
    An operation to represent binary div.
  }];
}

def RTL_MulOp : RTL_BinaryArithmeticOp<"mul", [Commutative]> {
  let summary = [{
    Mul op.
  }];

  let description = [{
    An operation to represent binary mul.
  }];
}

def RTL_PowOp : RTL_BinaryArithmeticOp<"pow"> {
  let summary = [{
    Pow op.
  }];

  let description = [{
    An operation to represent binary pow.
  }];
}

def RTL_RemOp : RTL_BinaryArithmeticOp<"rem"> {
  let summary = [{
    Rem op.
  }];

  let description = [{
    An operation to represent binary rem.
  }];
}

def RTL_ShiftLeftOp : RTL_BinaryArithmeticOp<"shift_left"> {
  let summary = [{
    Shift left op.
  }];

  let description = [{
    An operation to represent binary shift left.
  }];
}

def RTL_ShiftRightOp : RTL_BinaryArithmeticOp<"shift_right"> {
  let summary = [{
    Shift right op.
  }];

  let description = [{
    An operation to represent binary shift right.
  }];
}

def RTL_SubOp : RTL_BinaryArithmeticOp<"sub"> {
  let summary = [{
    Sub op.
  }];

  let description = [{
    An operation to represent binary sub.
  }];
}

// logical

class RTL_UnaryLogicalOp<string mnemonic, list<OpTrait> traits = []>
    : RTL_Op<mnemonic, !listconcat(traits, [NoSideEffect, SameOperandsAndResultType])> {
  let arguments = (ins
    RTL_ControlType:$arg
  );

  let results = (outs
    RTL_ControlType:$result
  );
}

def RTL_NotOp : RTL_UnaryLogicalOp<"not"> {
  let summary = [{
    Not op.
  }];

  let description = [{
    An operation to represent unary not.
  }];
}

class RTL_BinaryLogicalOp<string mnemonic, list<OpTrait> traits = []>
    : RTL_Op<mnemonic, !listconcat(traits, [NoSideEffect, Commutative, SameOperandsAndResultType])> {
  let arguments = (ins
    RTL_DataType:$left,
    RTL_DataType:$right
  );

  let results = (outs
    RTL_DataType:$result
  );
}

def RTL_AndOp : RTL_BinaryLogicalOp<"and"> {
  let summary = [{
    And op.
  }];

  let description = [{
    An operation to represent binary and.
  }];
}

def RTL_OrOp : RTL_BinaryLogicalOp<"or"> {
  let summary = [{
    Or op.
  }];

  let description = [{
    An operation to represent binary or.
  }];
}

def RTL_XorOp : RTL_BinaryLogicalOp<"xor"> {
  let summary = [{
    Xor op.
  }];

  let description = [{
    An operation to represent binary xor.
  }];
}

// structural ops
def RTL_ModuleOp
    : RTL_Op<"module", [FunctionLike, Symbol, SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = [{
    Module op.
  }];

  let description = [{
    An operation to represent a module in structural HDL.
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    TypeAttr:$type
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let extraClassDeclaration = [{
    // Hooks for OpTrait::FunctionLike
    unsigned getNumFuncArguments();
    unsigned getNumFuncResults();
    LogicalResult verifyType();
  }];
}

def RTL_ReturnOp
    : RTL_Op<"return", [NoSideEffect, HasParent<"ModuleOp">, Terminator]> {
  let summary = [{
    Return op.
  }];

  let description = [{
    An operation to terminate a module and indicate its return signals.
  }];

  let arguments = (ins
    Variadic<RTL_DataType>:$args
  );
}

def RTL_WireOp : RTL_Op<"wire", [ConstantLike]> {
  let summary = [{
    Wire op.
  }];

  let description = [{
    An operation to represent a wire in structural HDL.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$name
  );

  let results = (outs
    RTL_DataType:$value
  );
}

def RTL_InstanceOp : RTL_Op<"instance"> {
  let summary = [{
    Instance op.
  }];

  let description = [{
    An operation to represent an instance in structural HDL.
  }];

  let arguments = (ins
    SymbolRefAttr:$instantiation
  );
}

#endif // RTL_OPS
