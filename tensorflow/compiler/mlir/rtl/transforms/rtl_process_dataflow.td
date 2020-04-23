include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/rtl/ir/rtl_ops.td"
include "tensorflow/compiler/mlir/dataflow/ir/dataflow_ops.td"
include "tensorflow/compiler/mlir/xla/ir/hlo_ops.td"

// constant op is a direct rewrite
def ProcessHLOConstant : Pat<(HLO_ConstOp ElementsAttr:$value), (RTL_ConstantOp $value)>;

// arithmetic unary ops that are direct rewrites from HLO
foreach pair = [
  [HLO_NegOp, RTL_NegateOp]] in
  def : Pat<(pair[0] RTL_DataType:$arg), (pair[1] $arg)>;

// arithmetic binary ops that are direct rewrites from HLO
foreach pair = [
  [HLO_AddOp, RTL_AddOp],
  [HLO_DivOp, RTL_DivOp],
  [HLO_MulOp, RTL_MulOp],
  [HLO_PowOp, RTL_PowOp],
  [HLO_RemOp, RTL_RemOp],
  [HLO_ShiftLeftOp, RTL_ShiftLeftOp],
  [HLO_ShiftRightArithmeticOp, RTL_ShiftRightOp],
  [HLO_ShiftRightLogicalOp, RTL_ShiftRightOp],
  [HLO_SubOp, RTL_SubOp]] in
  def : Pat<(pair[0] RTL_DataType:$left, RTL_DataType:$right, $_), (pair[1] $left, $right)>;

// logical unary ops that are direct rewrites from HLO
foreach pair = [
  [HLO_NotOp, RTL_NotOp]] in
  def : Pat<(pair[0] RTL_ControlType:$arg), (pair[1] $arg)>;

// logical binary ops that are direct rewrites from HLO
foreach pair = [
  [HLO_AndOp, RTL_AndOp],
  [HLO_OrOp, RTL_OrOp],
  [HLO_XorOp, RTL_XorOp]] in
  def : Pat<(pair[0] RTL_ControlType:$left, RTL_ControlType:$right, $_), (pair[1] $left, $right)>;
