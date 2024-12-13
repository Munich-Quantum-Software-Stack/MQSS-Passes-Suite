/* This code and any associated documentation is provided "as is"

 IN NO EVENT SHALL LEIBNIZ-RECHENZENTRUM (LRZ) BE LIABLE TO ANY PARTY FOR
 DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT
 OF THE USE OF THIS CODE AND ITS DOCUMENTATION, EVEN IF LEIBNIZ-RECHENZENTRUM
 (LRZ) HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 THE AFOREMENTIONED EXCLUSIONS OF LIABILITY DO NOT APPLY IN CASE OF INTENT
 BY LEIBNIZ-RECHENZENTRUM (LRZ).

 LEIBNIZ-RECHENZENTRUM (LRZ), SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 FOR A PARTICULAR PURPOSE.

 THE CODE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, LEIBNIZ-RECHENZENTRUM (LRZ)
 HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 MODIFICATIONS.
-------------------------------------------------------------------------
  @author Martin Letras
  @date   December 2024
  @version 1.0
  @ brief
  CustomExamplePassPlugin
  Example pass to show how to implement a custo QUAKE MLIR pass.
  This pass performs an OpRewritePattern on Hadamard gates (quake::HOp).
  It just replaces all the ocurrences of quake::HOp by quake::SOp (S gate)
  in any given Quantum kernel.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/


#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Passes.hpp"

// Here is an example MLIR Pass that one can write externally and
// use via the cudaq-opt tool, with the --load-cudaq-plugin flag.
// The pass here is simple, replace Hadamard operations with S operations.

using namespace mlir;

namespace {

struct ReplaceH : public OpRewritePattern<quake::HOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(quake::HOp hOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<quake::SOp>(
        hOp, hOp.isAdj(), hOp.getParameters(), hOp.getControls(),
        hOp.getTargets());
    return success();
  }
};

class CustomExamplePassPlugin
    : public PassWrapper<CustomExamplePassPlugin, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CustomExamplePassPlugin)

  llvm::StringRef getArgument() const override { return "cudaq-custom-pass"; }

  void runOnOperation() override {
    auto circuit = getOperation();
    auto ctx = circuit.getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceH>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addIllegalOp<quake::HOp>();
    if (failed(applyPartialConversion(circuit, target, std::move(patterns)))) {
      circuit.emitOpError("simple pass failed");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCustomExamplePass(){
  return std::make_unique<CustomExamplePassPlugin>();
}
