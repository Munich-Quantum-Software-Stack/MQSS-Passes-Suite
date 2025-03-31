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

Z⋅H = H⋅X
*************************************************************************/

#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/SwitchOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_SWITCHZH
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class SwitchZH : public BaseMQSSPass<SwitchZH> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwitchZH)

  llvm::StringRef getArgument() const override { return "SwitchZH"; }
  llvm::StringRef getDescription() const override {
    return "Pass that switches a pattern composed by Z and Hadamard to "
           "Hadamard and X";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      patternSwitch<quake::ZOp, quake::HOp, quake::HOp, quake::XOp>(op);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createSwitchZHPass() {
  return std::make_unique<SwitchZH>();
}
