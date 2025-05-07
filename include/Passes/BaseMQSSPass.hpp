/* This code and any associated documentation is provided "as is"

Copyright 2025 Munich Quantum Software Stack Project

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
-------------------------------------------------------------------------
  author Martin Letras
  date   April 2025
  version 1.0
  brief
  This is the base class of the passes of the MQSS. It abstracts the parallel
  application of passes in case the given module contains more than one quantum
  circuit

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

// Base class extending PassWrapper with a common method
template <typename DerivedT>
class BaseMQSSPass
    : public PassWrapper<DerivedT, OperationPass<mlir::ModuleOp>> {
public:
  virtual void operationsOnQuantumKernel(
      func::FuncOp kernel) = 0; // this has to be re-written by each pass
private:
  std::tuple<SmallVector<Operation *, 16>, mlir::WalkResult>
  getQuakeKernels(mlir::ModuleOp module) {
    SmallVector<Operation *, 16> kernels;
    auto walkResult = module.walk([&kernels](Operation *op) {
      // Check if it is a quantum kernel
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (funcOp->hasAttr(cudaq::entryPointAttrName)) {
          kernels.push_back(funcOp);
          return WalkResult::advance();
        }
        for (auto arg : funcOp.getArguments())
          if (isa<quake::RefType, quake::VeqType>(arg.getType())) {
            kernels.push_back(funcOp);
            return WalkResult::advance();
          }
        // Skip functions which are not quantum kernels
        return WalkResult::skip();
      }
      // Check if it is controlled quake.apply
      if (auto applyOp = dyn_cast<quake::ApplyOp>(op))
        if (!applyOp.getControls().empty())
          return WalkResult::interrupt();

      return WalkResult::advance();
    });
    return std::make_tuple(kernels, walkResult);
  }

  void runOnOperation() override {
    auto module = this->getOperation();
    auto [kernels, walkResult] = getQuakeKernels(module);
    if (walkResult.wasInterrupted()) {
      module.emitError("Basis conversion doesn't work with `quake.apply`");
      this->signalPassFailure();
      return;
    }
    if (kernels.empty())
      return;
    // Process kernels in parallel
    parallelForEach(module.getContext(), kernels, [this](Operation *kernel) {
      if (auto funcOp = dyn_cast<func::FuncOp>(kernel)) {
        static_cast<DerivedT *>(this)->operationsOnQuantumKernel(funcOp);
      }
    });
  }
};
