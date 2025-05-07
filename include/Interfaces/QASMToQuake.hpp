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
******************************************************************************
  author Martin Letras
  date   May 2025
  version 1.0
******************************************************************************/
/** @file
 * @brief
 * @details This header defines a set of functions utilized to parse QASM
 * circuits to MLIR/Quake modules.
 *
 * @par
 * This header file is used by the QASM3ToQuakePass to perform the conversion of
 * QASM programs to MLIR/Quake modules.
 */

#pragma once

#include "Constants.hpp"
#include "ir/parsers/qasm3_parser/Parser.hpp"
#include "ir/parsers/qasm3_parser/Statement.hpp"
#include "ir/parsers/qasm3_parser/Types.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
/**
 * @typedef QASMVectorToQuakeVector
 * @brief The `QASMVectorToQuakeVector` is a map of type
 * `std::unordered_map<std::string, mlir::Value>`. The `key` is a string
 * corresponding to a quantum vector declared in the QASM program and the
 * `value` corresponds to an `mlir::Value` associated to a created and inserted
 * `quake::veq`.
 */
using QASMVectorToQuakeVector = std::unordered_map<std::string, mlir::Value>;

/**
 * @typedef QuantumVectorsOrder
 * @brief The `QuantumVectorsOrder` is list of pairs. Each entry of the pair is
 * composed of a string that is the id in the AST associated to the quantum
 * vector and the `int` value is the number of qubits in the register.
 */
using QuantumVectorsOrder = std::vector<std::pair<std::string, int>>;

namespace mqss::interfaces {

/**
 * @brief Given a gate type as a string, this functions inserts the
 corresponding Quake gate in the given builder.
   @details This method inserts a quake operation into an MLIR module associated
 with the builder passed as parameter.
    @param[in] gateId is a string specifying the type of the quantum gate to be
 inserted.
    @param[out] builder is an `OpBuilder` object associated with a MLIR module.
 It is used to insert new instructions to the corresponding MLIR module.
    @param[in] loc is the location of the new inserted instruction.
    @param[in] vecParams is the vector of arguments of the gate to be inserted,
 e.g., rotation angles in rotation gates.
    @param[in] vecControls is the vector of control qubits.
    @param[in] vecTargets is the vector of target qubits.
    @param[in] adj determines is the gate is an adjoint operation.
*/
void insertQASMGateIntoQuakeModule(std::string gateId, OpBuilder &builder,
                                   Location loc,
                                   std::vector<mlir::Value> vecParams,
                                   std::vector<mlir::Value> vecControls,
                                   std::vector<mlir::Value> vecTargets,
                                   bool adj);

/**
 * @brief Given a gate type as a string, this functions checks if the given gate
 type has control outputs.
   @details
    @param[in] gateType is a string specifying the type of a quantum gate.
    @return `true` if a gate is a multi-qubit gate with implicit controls.
*/
bool isMultiQubitGate(const std::string &gateType);

/**
 * @brief Given a gate type as a string, this functions returns the number of
 control outputs associated with the given gate type.
   @details
    @param[in] gateType is a string specifying the type a quantum gate.
    @return the number of control outputs of the give gate type.
*/
size_t getNumControls(const std::string &gateType);

/**
 * @brief Function that evaluates a numeric expression in the AST.
   @details
    @param[in] expr is as qasm3 numeric expression.
    @return a double value associated with the input numeric expression.
*/
double evaluateExpression(const std::shared_ptr<qasm3::Expression> &expr);

/**
 * @brief This function inserts a gate into a MLIR/Quake module.
   @details The QASM to Quake parser invokes this function to insert each gate
 in the AST.
    @param[in] gateCall is qasm3 gate declaration.
    @param[out] builder is an `OpBuilder` object associated with a MLIR module.
 It is used to insert new instructions (quantum registers) to the corresponding
 MLIR module.
    @param[in] loc is the location of the new inserted instruction (quantum
 registers).
    @param[in] inOp is the `return` operation in the module associated to the
 builder. New instructions are inserted before the`return` operation.
    @param[in] QASMToVectors map that returns a `mlir::Value` associated to a
 declared quantum vector, given the QASM quantum vector declaration.
*/
void insertGate(const std::shared_ptr<qasm3::GateCallStatement> &gateCall,
                OpBuilder &builder, Location loc, mlir::Operation *inOp,
                QASMVectorToQuakeVector QASMToVectors);

/**
 * @brief This function inserts measurements into a MLIR/Quake module.
   @details The QASM to Quake parser invokes this function to insert
 measurements into a given MLIR/Quake module.
    @param[in] statements is qasm3 measurements declarations.
    @param[out] builder is an `OpBuilder` object associated with a MLIR module.
 It is used to insert new instructions (quantum registers) to the corresponding
 MLIR module.
    @param[in] loc is the location of the new inserted instruction (quantum
 registers).
    @param[in] inOp is the `return` operation in the module associated to the
 builder. New instructions are inserted before the`return` operation.
    @param[in] QASMToVectors map that returns a `mlir::Value` associated to a
 declared quantum vector, given the QASM quantum vector declaration.
*/
void parseAndInsertMeasurements(
    const std::vector<std::shared_ptr<qasm3::Statement>> &statements,
    OpBuilder &builder, Location loc, mlir::Operation *inOp,
    QASMVectorToQuakeVector QASMToVectors);

/**
 * @brief This function return the quantum vectors and its order, given a
 Abstract Syntax Tree of a QASM program.
   @details This function receives an AST of QASM program, and create a
 `quake::veq` for each quantum register declared in the QASM program. Moreover,
 returns the order of appearance of each quantum register.
    @param[in] program is the AST of a QASM program.
    @param[out] builder is an `OpBuilder` object associated with a MLIR module.
 It is used to insert new instructions (quantum registers) to the corresponding
 MLIR module.
    @param[in] loc is the location of the new inserted instruction (quantum
 registers).
    @param[in] inOp is the `return` operation in the module associated to the
 builder. New instructions are inserted before the`return` operation.
    @return a tuple composed of `QASMVectorToQuakeVector` and
 `std::vector<std::pair<std::string, int>>`. The `QASMVectorToQuakeVector` is a
 map of type `std::unordered_map<std::string, mlir::Value>`. The `key` is a
 string corresponding to a quantum vector declared in the QASM program and the
 `value` corresponds to an `mlir::Value` associated to a created and inserted
 `quake::veq`. The second argument of the tuple is a vector that preserves the
 order of the inserted `quake::veq`, each entry is pair storing the quantum
 vector id, and the size of the quantum vector.
*/
std::tuple<QASMVectorToQuakeVector, QuantumVectorsOrder> insertAllocatedQubits(
    const std::vector<std::shared_ptr<qasm3::Statement>> &program,
    OpBuilder &builder, Location loc, mlir::Operation *inOp);
} // namespace mqss::interfaces
