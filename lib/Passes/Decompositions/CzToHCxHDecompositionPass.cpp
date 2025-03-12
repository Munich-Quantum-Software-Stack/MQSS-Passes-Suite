/* This code and any associated documentation is provided "as is"

Copyright 2024 Munich Quantum Software Stack Project

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

TODO: URL LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*************************************************************************
  author Martin Letras
  date   December 2024
  version 1.0

Adapted from:
https://quantumcomputing.stackexchange.com/questions/12458/show-that-a-cz-gate-can-be-implemented-using-a-cnot-gate-and-hadamard-gates

*************************************************************************/

#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct ReplaceCzToHCxH : public OpRewritePattern<quake::ZOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(quake::ZOp czOp,
                                PatternRewriter &rewriter) const override {
    if (czOp.getControls().size() != 1 && czOp.getTargets().size() != 1)
      return success(); // if the cx operation is not two qubit then do nothing
    // Get the operands of the XOp (control and target qubits)
    Value control = czOp.getControls()[0];
    Value target = czOp.getTargets()[0];
    Location loc = czOp.getLoc();
    // TODO: how to pass hOp.isAdj(), hOp.getParameters()
    // Insert the H gate on the original target qubit
    rewriter.create<quake::HOp>(loc, target);
    // Insert the Z gate swapping the control and target qubits
    rewriter.create<quake::XOp>(loc, control, target);
    // Insert the H gate on the original target qubit
    rewriter.create<quake::HOp>(loc, target);
    // Erase the original Cz operation
    rewriter.eraseOp(czOp);
    return success();
  }
};

class CzToHCxHDecompositionPass
    : public PassWrapper<CzToHCxHDecompositionPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CzToHCxHDecompositionPass)

  llvm::StringRef getArgument() const override {
    return "decomposition-cz-to-hcxh";
  }
  llvm::StringRef getDescription() const override {
    return "Decomposition pass of Cz by H, Cx, and H";
  }

  void runOnOperation() override {
    auto circuit = getOperation();
    auto ctx = circuit.getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceCzToHCxH>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addIllegalOp<quake::ZOp>();
    if (failed(applyPartialConversion(circuit, target, std::move(patterns)))) {
      circuit.emitOpError("CzToHCxHDecompositionPass failed");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCzToHCxHDecompositionPass() {
  return std::make_unique<CzToHCxHDecompositionPass>();
}
