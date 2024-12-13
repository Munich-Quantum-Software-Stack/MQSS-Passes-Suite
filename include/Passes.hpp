/* This code and any associated documentation is provided "as is"

 IN NO EVENT SHALL LEIBNIZ-RECHENZENTRUM (LRZ) BE LIABLE TO ANY PARTY FOR
 DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT
 OF THE USE OF THIS CODE AND ITS DOCUMENTATION, EVEN IF LEIBNIZ-RECHENZENTRUM
 (LRZ) HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 THE AFOREMENTIONED EXCLUSIONS OF LIABILITY DO NOT APPLY IN CASE OF INTENT
 BY LEIBNIZ-RECHENZENTRUM (LRZ).

 LEIBNIZ-RECHENZENTRUM (LRZ), SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 FOR A PARTICULAR PURPOSE.

 THE CODE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, LEIBNIZ-RECHENZENTRUM (LRZ)
 HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 MODIFICATIONS.
-------------------------------------------------------------------------
  @author Martin Letras
  @date   December 2024
  @version 1.0
  @ brief
  Header file that defines the signature for each QUAKE MLIR defined into the
  Munich Quantum Software Stack (MQSS).

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#ifndef PASSES_H
#define PASSES_H

#pragma once

#include <stdexcept>
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "qdmi.h"
#include "sc/heuristic/HeuristicMapper.hpp"

#define CUDAQ_PREFIX_FUNCTION "__nvqpp__mlirgen__"

namespace mqss::opt{

std::unique_ptr<mlir::Pass> createCustomExamplePass();
std::unique_ptr<mlir::Pass> createPrintQuakeGatesPass(llvm::raw_string_ostream &ostream);

std::unique_ptr<mlir::Pass> createQuakeQMapPass(Architecture &architecture, const Configuration &settings);

} // end namespace
#endif // PASSES_H
