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
  date   January 2025
  version 1.0

Adapted from: https://dl.acm.org/doi/10.5555/1972505

*************************************************************************/

#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cmath>
#include <numbers>

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_REGISTRATION
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;

namespace {

void normalizeAngleOfRotations(mlir::Operation *currentOp, OpBuilder builder) {
  if (!isa<quake::RxOp>(currentOp) && !isa<quake::RyOp>(currentOp) &&
      !isa<quake::RzOp>(currentOp))
    return; // do nothing if it is not rotation
  auto gate = dyn_cast<quake::OperatorInterface>(currentOp);
  double pi = std::numbers::pi;
  std::vector<mlir::Value> nParameters = {};
  mlir::IRRewriter rewriter(gate->getContext());
  for (auto parameter : gate.getParameters()) {
    double param =
        supportQuake::extractDoubleArgumentValue(parameter.getDefiningOp());
    param =
        param - (std::floor(param / (2 * pi)) * 2 * pi); // normalize the angle
    nParameters.push_back(
        supportQuake::createFloatValue(builder, gate.getLoc(), param));
  }
  ValueRange normParameters(nParameters);
  rewriter.setInsertionPointAfter(gate);
  if (isa<quake::RxOp>(gate))
    auto newGate = rewriter.create<quake::RxOp>(
        gate.getLoc(), gate.isAdj(), normParameters, gate.getControls(),
        gate.getTargets());
  if (isa<quake::RyOp>(gate))
    auto newGate = rewriter.create<quake::RyOp>(
        gate.getLoc(), gate.isAdj(), normParameters, gate.getControls(),
        gate.getTargets());
  if (isa<quake::RzOp>(gate))
    auto newGate = rewriter.create<quake::RzOp>(
        gate.getLoc(), gate.isAdj(), normParameters, gate.getControls(),
        gate.getTargets());
  rewriter.eraseOp(gate);
}

class NormalizeArgAnglePass
    : public PassWrapper<NormalizeArgAnglePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeArgAnglePass)

  llvm::StringRef getArgument() const override { return "NormalizeArgAngle"; }
  llvm::StringRef getDescription() const override {
    return "Optimization pass that normalizes the angle of Rx, Ry and Rz "
           "rotations";
  }

  void runOnOperation() override {
    auto circuit = getOperation();
    OpBuilder builder(&circuit.getBody());
    circuit.walk(
        [&](Operation *op) { normalizeAngleOfRotations(op, builder); });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mqss::opt::createNormalizeArgAnglePass() {
  return std::make_unique<NormalizeArgAnglePass>();
}
