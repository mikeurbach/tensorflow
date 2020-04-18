#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/dataflow/ir/dataflow_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace dataflow {

#include "tensorflow/compiler/mlir/dataflow/transforms/dataflow_legalize_hlo.cc.inc"

template<typename OpTy>
class WrapUnitRateOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter)
      const override {
    Operation *operation = op.getOperation();

    // create a new unit rate op
    auto unitRateOp = rewriter.create<UnitRateOp>(
        op.getLoc(), op.getType(), operation->getOperands());

    // wrap the inner op within the unit rate's body
    {
      OpBuilder::InsertionGuard guard(rewriter);

      Block *block = rewriter.createBlock(&unitRateOp.body());

      Location loc = unitRateOp.body().getLoc();

      auto innerOp = rewriter.create<OpTy>(
          loc, op.getType(), operation->getOperands(), op.getAttrs());

      rewriter.create<ReturnOp>(loc, innerOp.getResult());
    }

    // insert the unit rate op where the current op is
    rewriter.replaceOp(op, unitRateOp.getResults());

    return success();
  }
};

class LowerSelectOp : public OpRewritePattern<xla_hlo::SelectOp> {
 public:
  using OpRewritePattern<xla_hlo::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xla_hlo::SelectOp op, PatternRewriter &rewriter)
      const override {
    SmallVector<Value, 2> choices({op.on_false(), op.on_true()});

    auto mux = rewriter.create<MuxOp>(op.getLoc(), op.getType(), op.pred(), choices);

    rewriter.replaceOp(op, mux.getResult());

    return success();
  }
};

class LowerWhileOp : public OpRewritePattern<xla_hlo::WhileOp> {
  using OpRewritePattern<xla_hlo::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xla_hlo::WhileOp op, PatternRewriter &rewriter)
      const override {
    // create a new loop op
    auto loop = rewriter.create<LoopOp>(op.getLoc(), op.getType(), op.getOperand());

    // build the loop body
    {
      OpBuilder::InsertionGuard guard(rewriter);

      Location loc = loop.body().getLoc();

      // pick out the types returned by the conditional and body
      Type selectType = op.cond().front().back().getOperand(0).getType();
      Type valueType = loop.getOperand().getType();

      // create the block
      Block *block = rewriter.createBlock(&loop.body());
      block->addArguments({selectType, valueType});

      // put an initial value on the selection signal to select the input
      auto initialSelect = rewriter.create<InitialOp>(
          loc, selectType, block->getArgument(0),
          rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      // create a mux using the select signal to choose the init or loop value
      SmallVector<Value, 2> inputChoices(
          {loop.getOperand(), block->getArgument(1)});
      auto inputMux = rewriter.create<MuxOp>(
          loc, valueType, initialSelect, inputChoices);
      SmallVector<Value, 1> muxedInput({inputMux});

      // feed the muxed result into the condition and store the condition result
      Block *cond = &op.cond().front();
      Value condResult = (*std::next(cond->rbegin())).getResult(0);
      rewriter.eraseOp(&cond->back());
      rewriter.mergeBlocks(cond, block, muxedInput);

      // feed the muxed result into the body and store the body result
      Block *body = &op.body().front();
      Value bodyResult = (*std::next(body->rbegin())).getResult(0);
      rewriter.eraseOp(&body->back());
      rewriter.mergeBlocks(body, block, muxedInput);

      SmallVector<Type, 2> demuxResultType({valueType, valueType});

      // demux the input values into void if not selected
      auto inputDemux = rewriter.create<DemuxOp>(
          loc, demuxResultType, condResult, inputMux);
      auto inputChosen = inputDemux.getResult(0);
      auto inputDropped = inputDemux.getResult(1);
      rewriter.create<VoidOp>(loc, inputDropped);

      // demux the body result into void if not selected
      auto bodyDemux = rewriter.create<DemuxOp>(
          loc, demuxResultType, condResult, bodyResult);
      auto bodyDropped = bodyDemux.getResult(0);
      auto bodyChosen = bodyDemux.getResult(1);
      rewriter.create<VoidOp>(loc, bodyDropped);

      // stack a mux and demux to pass the chosen value to the right output
      SmallVector<Value, 2> resultChoices({inputChosen, bodyChosen});
      auto resultMux = rewriter.create<MuxOp>(
          loc, valueType, condResult, resultChoices);
      auto resultDemux = rewriter.create<DemuxOp>(
          loc, demuxResultType, condResult, resultMux);

      // return the cond result and the final demux's outputs
      SmallVector<Value, 3> returnResults(
          {condResult, resultDemux.getResult(1), resultDemux.getResult(0)});
      rewriter.create<ReturnOp>(loc, returnResults);
    }

    // insert the loop op where the current op is
    rewriter.replaceOp(op, loop.getResult());

    return success();
  }  
};

class LegalizeHLO : public PassWrapper<LegalizeHLO, FunctionPass> {
  void runOnFunction() override {
    Operation *op = getFunction();

    MLIRContext *context = op->getContext();

    ConversionTarget target(*context);

    // all of Dataflow is obviously legal
    target.addLegalDialect<DataflowDialect>();

    OwningRewritePatternList patterns;

    // register all patterns generated from tablegen
    populateWithGenerated(context, &patterns);

    // all of these ops can be implemented as a combinational circuit
    // so register patterns to wrap them in unit rate actors
    registerUnitRateWrapper<xla_hlo::AbsOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::CeilOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::ConstOp >(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::ConvertOp >(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::ClzOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::CosOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::ExpOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::Expm1Op>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::FloorOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::IsFiniteOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::LogOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::Log1pOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::NotOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::NegOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::PopulationCountOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::RoundOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::RsqrtOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::SignOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::SinOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::SqrtOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::TanhOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::AddOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::Atan2Op>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::DivOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::MaxOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::MinOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::MulOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::PowOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::RemOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::ShiftLeftOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::ShiftRightArithmeticOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::ShiftRightLogicalOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::SubOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::AndOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::OrOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::XorOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::CompareOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::TupleOp>(&target, &patterns, context);
    registerUnitRateWrapper<xla_hlo::GetTupleElementOp>(&target, &patterns, context);

    // convert select ops into mux ops
    patterns.insert<LowerSelectOp>(context);

    // convert while ops into loop ops
    patterns.insert<LowerWhileOp>(context);

    LogicalResult result = applyPartialConversion(op, target, patterns);

    if(failed(result)) {
      signalPassFailure();
    }
  }


 private:
  template <typename OpTy>
  static void registerUnitRateWrapper(ConversionTarget *target,
                                      OwningRewritePatternList *patterns,
                                      MLIRContext *context) {
    target->addDynamicallyLegalOp<OpTy>(isUnitRateWrapped);
    patterns->insert<WrapUnitRateOp<OpTy>>(context);
  }

  static bool isUnitRateWrapped(Operation *op) {
    return op->getParentOfType<UnitRateOp>().getOperation()->isProperAncestor(op);
  }
};

static PassRegistration<LegalizeHLO> pass(
    "dataflow-legalize-hlo", "Legalize from the XLA HLO dialect to the Dataflow dialect");

} // end namespace dataflow
} // end namespace mlir
