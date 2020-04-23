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

#endif // RTL_OPS
