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

Adapted from:  https://dl.acm.org/doi/10.5555/1972505

*************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Passes.hpp"
#include "Utils.hpp"

using namespace mlir;

namespace {

void ReplaceSZToSAdj(mlir::Operation *currentOp) {
  auto currentGate = dyn_cast_or_null<quake::ZOp>(*currentOp);
  if (!currentGate)
    return;
  // check single qubit h operation
  if(currentGate.getControls().size()!=0 && 
     currentGate.getTargets().size()!=1)
    return;
  // get previous
  auto prevOp = mqss::utils::getPreviousOperationOnTarget(currentGate, currentGate.getTargets()[0]);
  auto prevGate = dyn_cast_or_null<quake::SOp>(*prevOp);
  if (!prevGate)
    return;
  // check single qubit gate
  if(prevGate.getControls().size()!=0 && 
     prevGate.getTargets().size()!=1)
    return;
  if(prevGate.isAdj())
    return; 
  // I found the pattern, then I remove it from the circuit
  #ifdef DEBUG
    llvm::outs() << "Current Operation: ";
    currentGate->print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs() << "Previous Operation: ";
    prevGate->print(llvm::outs());
    llvm::outs() << "\n";
  #endif
  mlir::IRRewriter rewriter(currentGate->getContext());
  rewriter.setInsertionPointAfter(currentGate);
  rewriter.create<quake::SOp>(prevGate.getLoc(),
                              true,
                              prevGate.getParameters(),
                              prevGate.getControls(), 
                              prevGate.getTargets());
  rewriter.eraseOp(currentGate);
  rewriter.eraseOp(prevGate);
}
  
  class SToSAdjPass
      : public PassWrapper<SToSAdjPass , OperationPass<func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SToSAdjPass)
  
    llvm::StringRef getArgument() const override { return "s-z-to-s(adj)"; }
    llvm::StringRef getDescription() const override { return "Optimization pass that replaces a pattern composed of S and Z by S(adj)";}
  
    void runOnOperation() override {
      auto circuit = getOperation();
      circuit.walk([&](Operation *op){
        ReplaceSZToSAdj(op);
      });
    }
  };
} // namespace

std::unique_ptr<Pass> mqss::opt::createSToSAdjPass(){
  return std::make_unique<SToSAdjPass>();
}
