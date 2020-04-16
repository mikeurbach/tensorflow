#ifndef DATAFLOW_OPS
#define DATAFLOW_OPS

include "mlir/Interfaces/SideEffects.td"
include "mlir/IR/OpBase.td"

def Dataflow_Dialect : Dialect {
    let name = "dataflow";
    let cppNamespace = "dataflow";
}

class Dataflow_Op<string mnemonic, list<OpTrait> traits = []>
  : Op<Dataflow_Dialect, mnemonic, traits> {
}

def Dataflow_ControlType : AnyTypeOf<
    [AnySignlessInteger, TensorOf<[AnySignlessInteger]>]>;

def Dataflow_DataType : AnyTypeOf<
    [AnyStaticShapeTensor, TupleOf<[AnyStaticShapeTensor]>]>;

def Dataflow_VariadicDataType : Variadic<Dataflow_DataType>;

def Dataflow_UnitRateOp : Dataflow_Op<"unit_rate",
    [NoSideEffect, SingleBlockImplicitTerminator<"xla_hlo::ReturnOp">]> {
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

#endif // DATAFLOW_OPS
