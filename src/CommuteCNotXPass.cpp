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

mlir::Operation *getPreviousOperationOnTarget(mlir::Operation *currentOp, mlir::Value targetQubit){
  // Start from the previous operation
  mlir::Operation *prevOp = currentOp->getPrevNode();
  // Iterate through the previous operations in the block
  while (prevOp) {
    // Check if the operation has a target qubit and matches the given target
    if (auto quakeOp = dyn_cast<quake::OperatorInterface>(prevOp)) {
//      llvm::outs() << "Previous ";
//      quakeOp->print(llvm::outs());
//      llvm::outs() << "\n";
        int targetQCurr = mqss::utils::extractIndexFromQuakeExtractRefOp(targetQubit.getDefiningOp());
      for (mlir::Value target : quakeOp.getTargets()) {
        int targetQPrev = mqss::utils::extractIndexFromQuakeExtractRefOp(target.getDefiningOp());
        if (targetQCurr  == targetQPrev) 
          return prevOp;
      }
      for (mlir::Value control : quakeOp.getControls()) {
        int controlQPrev = mqss::utils::extractIndexFromQuakeExtractRefOp(control.getDefiningOp());
        if (targetQCurr  == controlQPrev) 
          return prevOp;
      }
    }
    // Move to the previous operation
    prevOp = prevOp->getPrevNode();
    }
    return nullptr; // No matching previous operation found
}

void CommuteCNotX(mlir::Operation *currentOp){
  //auto prevXOp = dyn_cast_or_null<quake::XOp>(prevOp);
  auto currXOp = dyn_cast_or_null<quake::XOp>(currentOp);
  if(!currXOp)
    return;
  // check single qubit CNot
  if(currXOp.getTargets().size()!=1 || currXOp.getControls().size()!=0)
    return;
  auto prevOp =getPreviousOperationOnTarget(currXOp, currXOp.getTargets()[0]);
  auto prevCNot = dyn_cast_or_null<quake::XOp>(prevOp);
  if (!prevCNot)
    return;
  // check that the previous is a two qubits XOp
  if (prevCNot.getControls().size() != 1 || prevCNot.getTargets().size() !=1)
    return;
  // check both targets are the same
  int targetCNot = mqss::utils::extractIndexFromQuakeExtractRefOp(prevCNot.getTargets()[0].getDefiningOp());
  int targetCurr = mqss::utils::extractIndexFromQuakeExtractRefOp(currXOp.getTargets()[0].getDefiningOp());
  if (targetCNot != targetCurr)
    return;
  #ifdef DEBUG
    llvm::outs() << "Current Operation: ";
    currXOp->print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs() << "Previous Operation: ";
    prevCNot->print(llvm::outs());
    llvm::outs() << "\n";
  #endif
  // At this point, I shoulb de able to do the commutation
  // Swap the two operations by cloning them in reverse order.
  mlir::IRRewriter rewriter(currXOp->getContext());
  rewriter.setInsertionPointAfter(currXOp);
  auto newCxOp = rewriter.create<quake::XOp>(prevCNot.getLoc(), prevCNot.isAdj(), 
                                            prevCNot.getParameters(), prevCNot.getControls(), 
                                            prevCNot.getTargets());
  rewriter.setInsertionPoint(newCxOp);
  rewriter.create<quake::XOp>(currXOp.getLoc(), currXOp.isAdj(), 
                              currXOp.getParameters(), currXOp.getControls(), 
                              currXOp.getTargets());

  // Erase the original operations
  rewriter.eraseOp(currXOp);
  rewriter.eraseOp(prevCNot);
}
/*


 : public OpRewritePattern<quake::XOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(quake::XOp cxOp,
                                PatternRewriter &rewriter) const override {
    if (cxOp.getControls().size() != 1 && cxOp.getTargets().size() !=1)
      return failure(); // if the cx operation is not two qubit then do nothing 
    
    // Check if the next operation is an RxOp. 
    // But may be the case that the next operation is quake.extract_ref.
    // I need to jump all the quake.extract_ref until I find a gate
    Operation *nextOp = cxOp->getNextNode();
    while(nextOp && isa<quake::ExtractRefOp>(nextOp))
      nextOp = nextOp->getNextNode();

    auto xOp = dyn_cast_or_null<quake::XOp>(nextOp);
    if(!xOp)
      return failure(); // if the next operation is not a rotation
    if(xOp.getTargets().size() != 1 && xOp.getControls().size() != 0)
      return failure(); // more paranoia checks
    
    #ifdef DEBUG
      llvm::outs() << "Next Operation (must be single qbit x): ";
      nextOp->print(llvm::outs());
      llvm::outs() << "\n";
      xOp.getTargets()[0].print(llvm::outs());
      llvm::outs() << "\n";
      xOp.getTargets()[0].print(llvm::outs());
      llvm::outs() << "\n";
    #endif

    int targetQubitX = mqss::utils::extractIndexFromQuakeExtractRefOp(xOp.getTargets()[0].getDefiningOp());
    int targetQubitCx = mqss::utils::extractIndexFromQuakeExtractRefOp(cxOp.getTargets()[0].getDefiningOp());
    // targets should be the same
    if((targetQubitX!=targetQubitCx) || (targetQubitX==-1 || targetQubitCx == -1) )
      return failure(); // return an do nothing
    // now I can commute the CNot and Rx
    // Swap the two operations by cloning them in reverse order.
    rewriter.setInsertionPointAfter(xOp);
    auto newCxOp = rewriter.create<quake::XOp>(cxOp.getLoc(), cxOp.isAdj(), 
                                              cxOp.getParameters(), cxOp.getControls(), 
                                              cxOp.getTargets());
    rewriter.setInsertionPoint(newCxOp);
    rewriter.create<quake::XOp>(xOp.getLoc(), xOp.isAdj(), 
                                xOp.getParameters(), xOp.getControls(), 
                                xOp.getTargets());

    // Erase the original operations
    rewriter.eraseOp(cxOp);
    rewriter.eraseOp(xOp);
    return success();
  }
};*/

class CommuteCNotXPass
    : public PassWrapper<CommuteCNotXPass , OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommuteCNotXPass)

  llvm::StringRef getArgument() const override { return "cudaq-custom-pass"; }

  void runOnOperation() override {
    auto circuit = getOperation();
    circuit.walk([&](Operation *op){
      CommuteCNotX(op);
      //llvm::outs() << "\n";
      //circuit->print(llvm::outs());
      //llvm::outs() << "\n";
    });
    /*
    auto ctx = circuit.getContext();
    llvm::outs() << "Starting to operate pass \n";
    RewritePatternSet patterns(ctx);
    patterns.insert<CommuteCNotX>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();

    // the applyPatternsAndFoldGreedily will try to attemp the replacement pattern
    // a given number of iterations
    // Configure the GreedyRewriteConfig.
    mlir::GreedyRewriteConfig config;
    config.maxIterations = 1; // Set the maximum number of iterations.
    applyPatternsAndFoldGreedily(circuit,  std::move(patterns), config);*/
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCommuteCNotXPass(){
  return std::make_unique<CommuteCNotXPass>();
}
