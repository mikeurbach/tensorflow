#ifndef DATAFLOW_OPS
#define DATAFLOW_OPS

include "mlir/Interfaces/SideEffects.td"
include "mlir/IR/OpBase.td"

// base

def Dataflow_Dialect : Dialect {
    let name = "dataflow";
    let cppNamespace = "dataflow";
}

class Dataflow_Op<string mnemonic, list<OpTrait> traits = []>
  : Op<Dataflow_Dialect, mnemonic, traits> {
}

// types

def Dataflow_ControlType : AnyTypeOf<
    [AnySignlessInteger, TensorOf<[AnySignlessInteger]>]>;

def Dataflow_DataType : AnyTypeOf<
    [AnyStaticShapeTensor, TupleOf<[AnyStaticShapeTensor]>]>;

def Dataflow_VariadicDataType : Variadic<Dataflow_DataType>;

// actor operations

def Dataflow_UnitRateOp : Dataflow_Op<"unit_rate",
    [NoSideEffect, SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = [{
    Unit-rate actor.
  }];

  let description = [{
    An operation that wraps a combinational op, connecting the inputs
    and outputs directly. Upon lowering, the relevant handshaking
    signals will be generated around the wrapped combinational circuit.
  }];

  let arguments = (ins
    Dataflow_VariadicDataType:$args
  );

  let results = (outs
    Dataflow_VariadicDataType:$results
  );

  let regions = (region
    SizedRegion<1>:$body
  );
}

def Dataflow_MuxOp : Dataflow_Op<"mux",
    [NoSideEffect]> {
  let summary = [{
    Mux actor.
  }];

  let description = [{
    An operation that uses a selection input to choose one of potentially
    several input choices to output. Waits for the selection to be valid,
    then waits for the selected input to be valid before outputting it.
  }];

  let arguments = (ins
    Dataflow_ControlType:$selection,
    Dataflow_VariadicDataType:$choices
  );

  let results = (outs
    Dataflow_DataType:$results
  );
}

def Dataflow_DemuxOp : Dataflow_Op<"demux",
    [NoSideEffect]> {
  let summary = [{
    Demux actor.
  }];

  let description = [{
    An operation that uses a selection input to send its input to one of
    potentially several output choices. Waits for the selection to be valid,
    then waits for the selected output to be ready before outputting it.
  }];

  let arguments = (ins
    Dataflow_ControlType:$selection,
    Dataflow_DataType:$arg
  );

  let results = (outs
    Dataflow_VariadicDataType:$results
  );
}

// helper operations

def Dataflow_ReturnOp : Dataflow_Op<"return",
  [Terminator]> {
  let summary = [{
    Return op.
  }];

  let summary = [{
    Terminates a block and designates one or more output channels
    for the current sub-circuit. The semantics of the operation
    determine how the returned channels are used.
  }];

  let arguments = (ins
    Dataflow_VariadicDataType:$results
  );
}

def Dataflow_VoidOp : Dataflow_Op<"void",
  [NoSideEffect]> {
  let summary = [{
    Void op.
  }];

  let summary = [{
    Consumes all of its inputs and produces no outputs. This op can
    be inserted to discard one or more tokens from the dataflow network.
  }];

  let arguments = (ins
    Dataflow_VariadicDataType:$args
  );
}

def Dataflow_InitialOp : Dataflow_Op<"initial",
  [NoSideEffect]> {
  let summary = [{
    Initial token op.
  }];

  let summary = [{
    Indicates that a channel should have an initial token placed on it.
    The op is otherwise transparent, and will be removed during lowering.
  }];

  let arguments = (ins
    Dataflow_ControlType:$arg,
    I1Attr:$value
  );

  let results = (outs
    Dataflow_ControlType:$result
  );
}

#endif // DATAFLOW_OPS
