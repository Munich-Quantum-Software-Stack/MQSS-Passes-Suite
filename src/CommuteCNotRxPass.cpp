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

Adapted from: https://quantumcomputing.stackexchange.com/questions/12458/show-that-a-cz-gate-can-be-implemented-using-a-cnot-gate-and-hadamard-gates

*************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Passes.hpp"
#include "Utils.hpp"

using namespace mlir;

namespace {

void CommuteCNotRx(mlir::Operation *currentOp){
  auto currRxOp = dyn_cast_or_null<quake::RxOp>(currentOp);
  if(!currRxOp)
    return;
  // check single qubit Rx
  if(currRxOp.getTargets().size()!=1 || currRxOp.getControls().size()!=0)
    return;
  auto prevOp = mqss::utils::getPreviousOperationOnTarget(currRxOp, currRxOp.getTargets()[0]);
  auto prevCNot = dyn_cast_or_null<quake::XOp>(prevOp);
  if (!prevCNot)
    return;
  // check that the previous is a two qubits XOp
  if (prevCNot.getControls().size() != 1 || prevCNot.getTargets().size() !=1)
    return;
  // check both targets are the same
  int targetCNot = mqss::utils::extractIndexFromQuakeExtractRefOp(prevCNot.getTargets()[0].getDefiningOp());
  int targetCurr = mqss::utils::extractIndexFromQuakeExtractRefOp(currRxOp.getTargets()[0].getDefiningOp());
  if (targetCNot != targetCurr)
    return;
  #ifdef DEBUG
    llvm::outs() << "Current Operation: ";
    currRxOp->print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs() << "Previous Operation: ";
    prevCNot->print(llvm::outs());
    llvm::outs() << "\n";
  #endif
  // At this point, I shoulb de able to do the commutation
  // Swap the two operations by cloning them in reverse order.
  mlir::IRRewriter rewriter(currRxOp->getContext());
  rewriter.setInsertionPointAfter(currRxOp);
  auto newCxOp = rewriter.create<quake::XOp>(prevCNot.getLoc(), prevCNot.isAdj(),
                                            prevCNot.getParameters(), prevCNot.getControls(),
                                            prevCNot.getTargets());
  rewriter.setInsertionPoint(newCxOp);
  rewriter.create<quake::RxOp>(currRxOp.getLoc(), currRxOp.isAdj(),
                              currRxOp.getParameters(), currRxOp.getControls(),
                              currRxOp.getTargets());
  // Erase the original operations
  rewriter.eraseOp(currRxOp);
  rewriter.eraseOp(prevCNot);
}

class CommuteCNotRxPass
    : public PassWrapper<CommuteCNotRxPass , OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommuteCNotRxPass)

  llvm::StringRef getArgument() const override { return "commute-cnotrx-pass"; }
  llvm::StringRef getDescription() const override { return "Apply commutation pass of the pattern CNot-Rx";}

  void runOnOperation() override {
    auto circuit = getOperation();
    circuit.walk([&](Operation *op){
      CommuteCNotRx(op);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCommuteCNotRxPass(){
  return std::make_unique<CommuteCNotRxPass>();
}
