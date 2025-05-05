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
-------------------------------------------------------------------------
  author Martin Letras
  date   May 2025
  version 1.0
  brief
  This header defines a set of functions utilized to parse QASM circuits to
  Quake/MLIR modules.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#pragma once

#include "Constants.hpp"
#include "ir/parsers/qasm3_parser/Parser.hpp"
#include "ir/parsers/qasm3_parser/Statement.hpp"
#include "ir/parsers/qasm3_parser/Types.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using QASMVectorToQuakeVector = std::unordered_map<std::string, mlir::Value>;

namespace mqss::interfaces {

// Given a gateId as a string, this functions inserts the corresponding Quake
// gate in the given builder. Parameters such as loc, vecParams (vector of
// arguments, i.e.,angles) vecControls and vecTargets must be given too.
void insertQASMGateIntoQuakeModule(std::string gateId, OpBuilder &builder,
                                   Location loc,
                                   std::vector<mlir::Value> vecParams,
                                   std::vector<mlir::Value> vecControls,
                                   std::vector<mlir::Value> vecTargets,
                                   bool adj);

// Function to determine if a gate is a multi-qubit gate with implicit controls
bool isMultiQubitGate(const std::string &gateType);

// Function to get the number of controls for a gate
size_t getNumControls(const std::string &gateType);

// This function returns the set of quantum registers declared in a given QASM
// program.
std::tuple<QASMVectorToQuakeVector, std::vector<std::pair<std::string, int>>>
insertAllocatedQubits(
    const std::vector<std::shared_ptr<qasm3::Statement>> &program,
    OpBuilder &builder, Location loc, mlir::Operation *inOp);

double evaluateExpression(const std::shared_ptr<qasm3::Expression> &expr);

// Function that inserts a QASM gate into a MLIR/Quake module
void insertGate(const std::shared_ptr<qasm3::GateCallStatement> &gateCall,
                OpBuilder &builder, Location loc, mlir::Operation *inOp,
                QASMVectorToQuakeVector QASMToVectors);

// Function that parses a given AST/QASM and inserts measurements into a
// MLIR/Quake
void parseAndInsertMeasurements(
    const std::vector<std::shared_ptr<qasm3::Statement>> &statements,
    OpBuilder &builder, Location loc, mlir::Operation *inOp,
    QASMVectorToQuakeVector QASMToVectors);

} // namespace mqss::interfaces
