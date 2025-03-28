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

bool checkDoublePiMultiplies(double angle) {
  const double pi = std::numbers::pi;
  const double doublePi = 2 * pi;
  if (std::fmod(angle, doublePi) == 0)
    return true;
  return false;
}

void nullRotationCancellation(mlir::Operation *currentOp) {
  if (!isa<quake::RxOp>(currentOp) && !isa<quake::RyOp>(currentOp) &&
      !isa<quake::RzOp>(currentOp))
    return; // do nothing if it is not rotation
  auto gate = dyn_cast<quake::OperatorInterface>(currentOp);
  // assuming that parameters are all rotation angles
  bool deleteGate = true;
  for (auto parameter : gate.getParameters()) {
    double param =
        supportQuake::extractDoubleArgumentValue(parameter.getDefiningOp());
    if (!checkDoublePiMultiplies(param) && param != 0)
      deleteGate = false;
  }
  if (deleteGate) {
    // remove the rotation gate
    mlir::IRRewriter rewriter(gate->getContext());
    rewriter.eraseOp(gate);
  }
}

class NullRotationCancellationPass
    : public PassWrapper<NullRotationCancellationPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NullRotationCancellationPass)

  llvm::StringRef getArgument() const override { return "CancelNullRotations"; }
  llvm::StringRef getDescription() const override {
    return "Optimization pass that removes of Rx, Ry and Rz null rotations";
  }

  void runOnOperation() override {
    auto circuit = getOperation();
    circuit.walk([&](Operation *op) { nullRotationCancellation(op); });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mqss::opt::createNullRotationCancellationPass() {
  return std::make_unique<NullRotationCancellationPass>();
}
