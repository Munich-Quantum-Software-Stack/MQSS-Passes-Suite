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

It applies the following transformations

X⋅H = H⋅Z
Y⋅H = H⋅Y
Z⋅H = H⋅X

*************************************************************************/

#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

void SwitchPauliH(mlir::Operation *currentOp) {
  auto currentGate = dyn_cast_or_null<quake::HOp>(*currentOp);
  if (!currentGate)
    return;
  // check single qubit h operation
  if (currentGate.getControls().size() != 0 ||
      currentGate.getTargets().size() != 1)
    return;
  // get previous
  auto prevOp = supportQuake::getPreviousOperationOnTarget(
      currentGate, currentGate.getTargets()[0]);
  if (!prevOp)
    return;
  if (!isa<quake::XOp>(prevOp) && !isa<quake::YOp>(prevOp) &&
      !isa<quake::ZOp>(prevOp))
    return; // if no pauli, do nothing
  auto prevGateInt = dyn_cast<quake::OperatorInterface>(prevOp);
  // check single qubit pauli operation
  if (prevGateInt.getControls().size() != 0 ||
      prevGateInt.getTargets().size() != 1)
    return;
// I found the pattern, then I remove it from the circuit
#ifdef DEBUG
  llvm::outs() << "Current Operation: ";
  currentGate->print(llvm::outs());
  llvm::outs() << "\n";
  llvm::outs() << "Previous Operation: ";
  prevGateInt->print(llvm::outs());
  llvm::outs() << "\n";
#endif
  mlir::IRRewriter rewriter(currentGate->getContext());
  rewriter.setInsertionPointAfter(currentGate);
  if (isa<quake::XOp>(prevOp)) {
    auto prevGate = dyn_cast_or_null<quake::XOp>(*prevOp);
    if (!prevGate)
      return;
    rewriter.create<quake::ZOp>(prevGate.getLoc(), prevGate.isAdj(),
                                prevGate.getParameters(),
                                prevGate.getControls(), prevGate.getTargets());
    rewriter.eraseOp(prevGate);
  } else if (isa<quake::YOp>(prevOp)) {
    auto prevGate = dyn_cast_or_null<quake::YOp>(*prevOp);
    if (!prevGate)
      return;
    rewriter.create<quake::YOp>(prevGate.getLoc(), prevGate.isAdj(),
                                prevGate.getParameters(),
                                prevGate.getControls(), prevGate.getTargets());
    rewriter.eraseOp(prevGate);
  } else if (isa<quake::ZOp>(prevOp)) {
    auto prevGate = dyn_cast_or_null<quake::ZOp>(*prevOp);
    if (!prevGate)
      return;
    rewriter.create<quake::XOp>(prevGate.getLoc(), prevGate.isAdj(),
                                prevGate.getParameters(),
                                prevGate.getControls(), prevGate.getTargets());
    rewriter.eraseOp(prevGate);
  }
}

class PauliGateAndHadamardSwitchPass
    : public PassWrapper<PauliGateAndHadamardSwitchPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PauliGateAndHadamardSwitchPass)

  llvm::StringRef getArgument() const override {
    return "switch-pauli-hadamard";
  }
  llvm::StringRef getDescription() const override {
    return "Pass that switches a pattern composed by {X,Y,Z} (Pauli) and "
           "Hadamard";
  }

  void runOnOperation() override {
    auto circuit = getOperation();
    circuit.walk([&](Operation *op) { SwitchPauliH(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createPauliGateAndHadamardSwitchPass() {
  return std::make_unique<PauliGateAndHadamardSwitchPass>();
}
