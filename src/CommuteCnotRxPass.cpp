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

struct CommuteCNotRx : public OpRewritePattern<quake::XOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(quake::XOp cxOp,
                                PatternRewriter &rewriter) const override {
    if (cxOp.getControls().size() != 1 && cxOp.getTargets().size() !=1)
      return success(); // if the cx operation is not two qubit then do nothing 
    
    // Check if the next operation is an RxOp. 
    // But may be the case that the next operation is quake.extract_ref.
    // I need to jump all the quake.extract_ref until I find a gate
    Operation *nextOp = cxOp->getNextNode();
    while(nextOp && isa<quake::ExtractRefOp>(nextOp))
      nextOp = nextOp->getNextNode();

    auto rxOp = dyn_cast_or_null<quake::RxOp>(nextOp);
    if(!rxOp)
      return success(); // if the next operation is not a rotation
    if(rxOp.getTargets().size() != 1 && rxOp.getControls().size() != 0)
      return success(); // more paranoia checks
    
    #ifdef DEBUG
      llvm::outs() << "Next Operation (must be rx): ";
      nextOp->print(llvm::outs());
      llvm::outs() << "\n";
      rxOp.getTargets()[0].print(llvm::outs());
      llvm::outs() << "\n";
      cxOp.getTargets()[0].print(llvm::outs());
      llvm::outs() << "\n";
    #endif

    int targetQubitRx = mqss::utils::extractIndexFromQuakeExtractRefOp(rxOp.getTargets()[0].getDefiningOp());
    int targetQubitCx = mqss::utils::extractIndexFromQuakeExtractRefOp(cxOp.getTargets()[0].getDefiningOp());
    // targets should be the same
    if((targetQubitRx!=targetQubitCx) || (targetQubitRx==-1 || targetQubitCx == -1) )
      return success(); // return an do nothing
    // now I can commute the CNot and Rx
    // Swap the two operations by cloning them in reverse order.
    rewriter.setInsertionPointAfter(rxOp);
    auto newXOp = rewriter.create<quake::XOp>(cxOp.getLoc(), cxOp.isAdj(), 
                                              cxOp.getParameters(), cxOp.getControls(), 
                                              cxOp.getTargets());
    rewriter.setInsertionPoint(newXOp);
    rewriter.create<quake::RxOp>(rxOp.getLoc(), rxOp.isAdj(), 
                                 rxOp.getParameters(), rxOp.getControls(), 
                                 rxOp.getTargets());

    // Erase the original operations
    rewriter.eraseOp(cxOp);
    rewriter.eraseOp(rxOp);
    return success();
  }
};

class CommuteCNotRxPass
    : public PassWrapper<CommuteCNotRxPass , OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommuteCNotRxPass)

  llvm::StringRef getArgument() const override { return "cudaq-custom-pass"; }

  void runOnOperation() override {
    auto circuit = getOperation();
    auto ctx = circuit.getContext();
    llvm::outs() << "Starting to operate pass \n";
    RewritePatternSet patterns(ctx);
    patterns.insert<CommuteCNotRx>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();

    // the applyPatternsAndFoldGreedily will try to attemp the replacement pattern
    // a given number of iterations
    // Configure the GreedyRewriteConfig.
    mlir::GreedyRewriteConfig config;
    config.maxIterations = 2; // Set the maximum number of iterations.
    applyPatternsAndFoldGreedily(circuit,  std::move(patterns), config);
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCommuteCNotRxPass(){
  return std::make_unique<CommuteCNotRxPass>();
}
