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

Adapted from: https://link.springer.com/chapter/10.1007/978-981-287-996-7_2

*************************************************************************/

#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_REGISTRATION
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;

namespace {

void commuteZCNot(mlir::Operation *currentOp) {
  auto currentGate = dyn_cast_or_null<quake::XOp>(*currentOp);
  if (!currentGate)
    return;
  // check that the current gate is compliant with the number of controls and
  // targets
  if (currentGate.getControls().size() != 1 ||
      currentGate.getTargets().size() != 1)
    return;
  // get the previous operation to check the swap pattern
  auto prevOp = supportQuake::getPreviousOperationOnTarget(
      currentGate, currentGate.getControls()[0]);
  if (!prevOp)
    return;
  auto previousGate = dyn_cast_or_null<quake::ZOp>(prevOp);
  if (!previousGate)
    return;
  // check that the previous gate is compliant with the number of controls and
  // targets
  if (previousGate.getControls().size() != 0 ||
      previousGate.getTargets().size() != 1)
    return; // check both targets are the same

  int targetPrev = supportQuake::extractIndexFromQuakeExtractRefOp(
      previousGate.getTargets()[0].getDefiningOp());
  int controlCurr = supportQuake::extractIndexFromQuakeExtractRefOp(
      currentGate.getControls()[0].getDefiningOp());
  if (targetPrev == controlCurr) {
// the pattern is:
// -|z|---.---   ---.----|z|-
//        |    =    |
// ------|x|--   --|x|-------
#ifdef DEBUG
    llvm::outs() << "Current Operation: ";
    currentGate->print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs() << "Previous Operation: ";
    previousGate->print(llvm::outs());
    llvm::outs() << "\n";
#endif
    // At this point, I should de able to do the commutation
    // Swap the two operations by cloning them in reverse order.
    mlir::IRRewriter rewriter(currentGate->getContext());
    rewriter.setInsertionPointAfter(currentGate);
    rewriter.create<quake::ZOp>(previousGate.getLoc(), previousGate.isAdj(),
                                previousGate.getParameters(),
                                previousGate.getControls(),
                                previousGate.getTargets());
    // Erase the original operations
    // rewriter.eraseOp(currentGate);
    rewriter.eraseOp(previousGate);
    return;
  }
}

class CommuteZCNotPass
    : public PassWrapper<CommuteZCNotPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommuteZCNotPass)

  llvm::StringRef getArgument() const override { return "CommuteZCx"; }
  llvm::StringRef getDescription() const override {
    return "Apply commutation pass to pattern Z-CNot to CNot-Z";
  }

  void runOnOperation() override {
    auto circuit = getOperation();
    circuit.walk([&](Operation *op) { commuteZCNot(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCommuteZCNotPass() {
  return std::make_unique<CommuteZCNotPass>();
}

// Register the pass on initialization
void registerCommuteZCNotPass() { ::registerCommuteZCNotPass(); }
