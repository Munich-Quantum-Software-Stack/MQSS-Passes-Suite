#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Here is an example MLIR Pass that one can write externally and
// use via the cudaq-opt tool, with the --load-cudaq-plugin flag.
// The pass here is simple, replace Hadamard operations with S operations.

using namespace mlir;

namespace {

//struct ReplaceH : public OpRewritePattern<quake::HOp> {
//  using OpRewritePattern::OpRewritePattern;
//  LogicalResult matchAndRewrite(quake::HOp hOp,
//                                PatternRewriter &rewriter) const override {
//    rewriter.replaceOpWithNewOp<quake::SOp>(
//        hOp, hOp.isAdj(), hOp.getParameters(), hOp.getControls(),
//        hOp.getTargets());
//    return success();
//  }
//};

class PrintQuakeGatesPass
    : public PassWrapper<PrintQuakeGatesPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintQuakeGatesPass)

  llvm::StringRef getArgument() const override { return "cudaq-custom-pass"; }

  void runOnOperation() override {
    auto circuit = getOperation();
    circuit.walk([](Operation *op){
      if (op->getDialect()->getNamespace() == "quake") {
        llvm::outs() << "Quantum Operation: " << op->getName().getStringRef() << "\n";

        // Iterate over the operands (qubits) the operation acts on
        for (Value operand : op->getOperands()) {
          if (operand.getType().isa<quake::RefType>()) { // Check if it's a qubit reference
            llvm::outs() << "  Acts on qubit: " << operand << "\n";
          }
        }

      }
    });

    //auto ctx = circuit.getContext();

    //RewritePatternSet patterns(ctx);
    //patterns.insert<ReplaceH>(ctx);
    //ConversionTarget target(*ctx);
    //target.addLegalDialect<quake::QuakeDialect>();
    //target.addIllegalOp<quake::HOp>();
    //if (failed(applyPartialConversion(circuit, target, std::move(patterns)))) {
    //  circuit.emitOpError("simple pass failed");
    //  signalPassFailure();
    //}
  }
};

} // namespace

CUDAQ_REGISTER_MLIR_PASS(PrintQuakeGatesPass)
