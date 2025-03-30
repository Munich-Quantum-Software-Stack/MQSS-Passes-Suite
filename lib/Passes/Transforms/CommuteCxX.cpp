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
  date   December 2024
  version 1.0

  Adapted from: https://link.springer.com/chapter/10.1007/978-981-287-996-7_2

*************************************************************************/

#include "Passes/Transforms.hpp"
#include "Support/Transforms/CommutateOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_COMMUTECXX
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class CommuteCxX : public PassWrapper<CommuteCxX, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommuteCxX)

  llvm::StringRef getArgument() const override { return "CommuteCxX"; }
  llvm::StringRef getDescription() const override {
    return "Apply commutation pass to pattern CNot-X";
  }

  void runOnOperation() override {
    auto circuit = getOperation();
    circuit.walk([&](Operation *op) {
      commuteOperation<quake::XOp, quake::XOp>(op, 1, 1, 0, 1);
      // CommuteCNotX(op);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCommuteCxXPass() {
  return std::make_unique<CommuteCxX>();
}
