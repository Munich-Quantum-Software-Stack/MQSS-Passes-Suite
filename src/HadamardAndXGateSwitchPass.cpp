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

H⋅X = Z⋅H

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

  class HadamardAndXGateSwitchPass
      : public PassWrapper<HadamardAndXGateSwitchPass, OperationPass<func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HadamardAndXGateSwitchPass)
  
    llvm::StringRef getArgument() const override { return "switch-hadamard-x"; }
    llvm::StringRef getDescription() const override { return "Pass that switches a pattern composed Hadamard and X to Z and Hadamard";}
  
    void runOnOperation() override {
      auto circuit = getOperation();
      circuit.walk([&](Operation *op){
        mqss::utils::patternSwitch<quake::HOp, quake::XOp, 
                                   quake::ZOp, quake::HOp>(op);
      });
    }
  };
} // namespace

std::unique_ptr<Pass> mqss::opt::createHadamardAndXGateSwitchPass(){
  return std::make_unique<HadamardAndXGateSwitchPass>();
}
