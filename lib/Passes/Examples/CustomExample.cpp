/* This code and any associated documentation is provided "as is"

Copyright 2024 Munich Quantum Software Stack Project

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://github.com/Munich-Quantum-Software-Stack/passes/blob/develop/LICENSE

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
*************************************************************************/

#include "Passes/Examples.hpp"
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
    : public PassWrapper<CustomExamplePassPlugin,
                         OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CustomExamplePassPlugin)

  llvm::StringRef getArgument() const override { return "cudaq-custom-pass"; }
  llvm::StringRef getDescription() const override {
    return "Apply dummy example pass that replaces each H gate by a S gate";
  }
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

std::unique_ptr<Pass> mqss::opt::createCustomExamplePass() {
  return std::make_unique<CustomExamplePassPlugin>();
}
