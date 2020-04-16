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

def Dataflow_UnitRateType : Variadic<AnyTypeOf<[AnyStaticShapeTensor, TupleOf<[AnyStaticShapeTensor]>]>>;

def Dataflow_UnitRateOp : Dataflow_Op<"unit_rate", [NoSideEffect]> {
  let summary = [{
    Unit-rate actor.
  }];

  let description = [{
    An operation that wraps a combinational op, connecting the inputs
    and outputs directly. Upon lowering, the relevant handshaking
    signals will be generated around the wrapped combinational circuit.
  }];

  let arguments = (ins
    Dataflow_UnitRateType:$args
  );

  let results = (outs
    Dataflow_UnitRateType:$results
  );

  let regions = (region
    SizedRegion<1>:$body
  );
}

#endif // DATAFLOW_OPS
