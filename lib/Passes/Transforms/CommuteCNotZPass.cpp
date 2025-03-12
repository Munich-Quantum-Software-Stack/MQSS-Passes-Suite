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

using namespace mlir;

namespace {

void commuteCNotZ(mlir::Operation *currentOp) {
  auto currentGate = dyn_cast_or_null<quake::ZOp>(*currentOp);
  if (!currentGate)
    return;
  // check that the current gate is compliant with the number of controls and
  // targets
  if (currentGate.getControls().size() != 0 ||
      currentGate.getTargets().size() != 1)
    return;
  // get the previous operation to check the swap pattern
  auto prevOp = supportQuake::getPreviousOperationOnTarget(
      currentGate, currentGate.getTargets()[0]);
  if (!prevOp)
    return;
  auto previousGate = dyn_cast_or_null<quake::XOp>(prevOp);
  if (!previousGate)
    return;
  // check that the previous gate is compliant with the number of controls and
  // targets
  if (previousGate.getControls().size() != 1 ||
      previousGate.getTargets().size() != 1)
    return; // check both targets are the same

  int targetPrev = supportQuake::extractIndexFromQuakeExtractRefOp(
      previousGate.getTargets()[0].getDefiningOp());
  int controlPrev = supportQuake::extractIndexFromQuakeExtractRefOp(
      previousGate.getControls()[0].getDefiningOp());
  int targetCurr = supportQuake::extractIndexFromQuakeExtractRefOp(
      currentGate.getTargets()[0].getDefiningOp());
  if (targetCurr == controlPrev) {
// the pattern is:
// ---.----|z|-     -|z|---.---
//    |          =         |
// --|x|-------     ------|x|--
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
    rewriter.create<quake::XOp>(previousGate.getLoc(), previousGate.isAdj(),
                                previousGate.getParameters(),
                                previousGate.getControls(),
                                previousGate.getTargets());
    // Erase the original operations
    // rewriter.eraseOp(currentGate);
    rewriter.eraseOp(previousGate);
    return;
  }
  /*if(targetCurr == targetPrev){
    // Target-Z non-commute
    // the pattern is:
    // ---.---------     -|z|---.---
    //    |           =         |
    // --|x|---|z|--     ------|x|--
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
    auto newGate = rewriter.create<quake::ZOp>(currentGate.getLoc(),
                                               currentGate.isAdj(),
                                               currentGate.getParameters(),
                                               currentGate.getControls(),
                                               // target = control previous
                                               previousGate.getControls()[0]);
    rewriter.setInsertionPointAfter(newGate);
    rewriter.create<quake::XOp>(previousGate.getLoc(),
                                previousGate.isAdj(),
                                previousGate.getParameters(),
                                previousGate.getControls(),
                                previousGate.getTargets());
    // Erase the original operations
    rewriter.eraseOp(currentGate);
    rewriter.eraseOp(previousGate);
    return;
  }*/
}

class CommuteCNotZPass
    : public PassWrapper<CommuteCNotZPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommuteCNotZPass)

  llvm::StringRef getArgument() const override { return "commute-cnotz-pass"; }
  llvm::StringRef getDescription() const override {
    return "Apply commutation pass of the pattern CNot-Z to Z-CNot";
  }

  void runOnOperation() override {
    auto circuit = getOperation();
    circuit.walk([&](Operation *op) { commuteCNotZ(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCommuteCNotZPass() {
  return std::make_unique<CommuteCNotZPass>();
}
