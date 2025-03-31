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

Adapted from:
https://agra.informatik.uni-bremen.de/doc/konf/2021_DSD_CNOTs_remote_gates.pdf

*************************************************************************/

#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_REVERSECX
#include "Passes/Decompositions.h.inc"
} // namespace mqss::opt
using namespace mlir;

namespace {

void ReverseCNot(mlir::Operation *currentOp) {
  auto cxOp = dyn_cast_or_null<quake::XOp>(*currentOp);
  if (!cxOp)
    return;
  if (cxOp.getControls().size() != 1 || cxOp.getTargets().size() != 1)
    return; // if the cx operation is not two qubit then do nothing
  // Get the operands of the XOp (control and target qubits)
  Value control = cxOp.getControls()[0];
  Value target = cxOp.getTargets()[0];
  Location loc = cxOp.getLoc();

  mlir::IRRewriter rewriter(cxOp->getContext());
  rewriter.setInsertionPointAfter(cxOp);
  // TODO: howt to pass hOp.isAdj(), hOp.getParameters()
  // Insert the H gates on control and target qubits
  rewriter.create<quake::HOp>(loc, target);
  rewriter.create<quake::HOp>(loc, control);
  // Insert the X gate swapping the control and target qubits
  rewriter.create<quake::XOp>(loc, target, control);
  // Insert the H gates after CNot
  rewriter.create<quake::HOp>(loc, target);
  rewriter.create<quake::HOp>(loc, control);
  // Erase the original Cx operation
  rewriter.eraseOp(cxOp);
}

class ReverseCx : public BaseMQSSPass<ReverseCx> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReverseCx)

  llvm::StringRef getArgument() const override { return "ReverseCx"; }
  llvm::StringRef getDescription() const override {
    return "Decomposition pass that reverses the control and targets of each "
           "two-qubits CNot gate in a circuit";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    kernel.walk([&](Operation *op) { ReverseCNot(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createReverseCxPass() {
  return std::make_unique<ReverseCx>();
}
