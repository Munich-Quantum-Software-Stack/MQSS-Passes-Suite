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

It applies the following transformations

X⋅H = H⋅Z
*************************************************************************/

#include "Passes/Transforms.hpp"
#include "Support/Transforms/SwitchOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_REGISTRATION
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class XGateAndHadamardSwitchPass
    : public PassWrapper<XGateAndHadamardSwitchPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XGateAndHadamardSwitchPass)

  llvm::StringRef getArgument() const override { return "SwitchXH"; }
  llvm::StringRef getDescription() const override {
    return "Pass that switches a pattern composed by X and Hadamard to "
           "Hadamard and Z";
  }

  void runOnOperation() override {
    auto circuit = getOperation();
    circuit.walk([&](Operation *op) {
      patternSwitch<quake::XOp, quake::HOp, quake::HOp, quake::ZOp>(op);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createXGateAndHadamardSwitchPass() {
  return std::make_unique<XGateAndHadamardSwitchPass>();
}

// Register the pass on initialization
void registerXGateAndHadamardSwitchPass() {
  ::registerXGateAndHadamardSwitchPass();
}
