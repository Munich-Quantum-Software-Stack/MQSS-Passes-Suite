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

void CommuteCNotZ(mlir::Operation *currentOp){
  auto currZOp = dyn_cast_or_null<quake::ZOp>(currentOp);
  if(!currZOp)
    return;
  // check single qubit Z
  if(currZOp.getTargets().size()!=1 || currZOp.getControls().size()!=0)
    return;
  auto prevOp = mqss::utils::getPreviousOperationOnTarget(currZOp, currZOp.getTargets()[0]);
  auto prevCNot = dyn_cast_or_null<quake::XOp>(prevOp);
  if (!prevCNot)
    return;
  // check that the previous is a two qubits XOp
  if (prevCNot.getControls().size() != 1 || prevCNot.getTargets().size() !=1)
    return;
  // check both targets are the same
  int targetCNot = mqss::utils::extractIndexFromQuakeExtractRefOp(prevCNot.getTargets()[0].getDefiningOp());
  int targetCurr = mqss::utils::extractIndexFromQuakeExtractRefOp(currZOp.getTargets()[0].getDefiningOp());
  if (targetCNot != targetCurr)
    return;
  #ifdef DEBUG
    llvm::outs() << "Current Operation: ";
    currZOp->print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs() << "Previous Operation: ";
    prevCNot->print(llvm::outs());
    llvm::outs() << "\n";
  #endif
  // At this point, I shoulb de able to do the commutation
  // Swap the two operations by cloning them in reverse order.
  mlir::IRRewriter rewriter(currZOp->getContext());
  rewriter.setInsertionPointAfter(currZOp);
  auto newCxOp = rewriter.create<quake::XOp>(prevCNot.getLoc(), prevCNot.isAdj(), 
                                            prevCNot.getParameters(), prevCNot.getControls(), 
                                            prevCNot.getTargets());
  rewriter.setInsertionPoint(newCxOp);
  rewriter.create<quake::ZOp>(currZOp.getLoc(), currZOp.isAdj(), 
                              currZOp.getParameters(), currZOp.getControls(), 
                              currZOp.getTargets());
  // Erase the original operations
  rewriter.eraseOp(currZOp);
  rewriter.eraseOp(prevCNot);
}

class CommuteCNotZPass
    : public PassWrapper<CommuteCNotZPass , OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommuteCNotZPass)

  llvm::StringRef getArgument() const override { return "commute-cnotz-pass"; }
  llvm::StringRef getDescription() const override { return "Apply commutation pass of the pattern CNot-Z to Z-CNot";}

  void runOnOperation() override {
    auto circuit = getOperation();
    circuit.walk([&](Operation *op){
      CommuteCNotZ(op);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCommuteCNotZPass(){
  return std::make_unique<CommuteCNotZPass>();
}
