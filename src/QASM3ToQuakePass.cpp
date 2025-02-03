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
-------------------------------------------------------------------------
  author Martin Letras
  date   Januray 2025
  version 1.0
  brief
  PrintQuakeGatesPass(llvm::raw_string_ostream ostream)
  Example MLIR pass that shows how to traverse a Quantum kernel written in
  QUAKE MLIR.
  The pass prints in ostream the type of each quantum gate and its operand(s)
  qubits.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <iomanip>
#include <regex>
#include "Passes.hpp"
#include "Utils.hpp"
#include "ir/parsers/qasm3_parser/Parser.hpp"

using namespace mlir;

  namespace {
  
  class QASM3ToQuakePass
      : public PassWrapper<QASM3ToQuakePass, OperationPass<func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QASM3ToQuakePass)
  
    QASM3ToQuakePass(std::istringstream &qasmStream) : qasmStream(qasmStream) {}
  
    llvm::StringRef getArgument() const override { return "convert-qasm3-to-quake"; }
    llvm::StringRef getDescription() const override { return "Convert QASM3 to Quake Operations"; };
  
    void runOnOperation() override {
      auto circuit = getOperation();
      // Get the function name
      StringRef funcName = circuit.getName();
      if (!(funcName.find(std::string(CUDAQ_PREFIX_FUNCTION)) != std::string::npos))
        return; // do nothing if the funcion is not cudaq kernel
    
    // Create the parser
    qasm3::Parser parser(&qasmStream,true);
    // Parse the program to get the AST
    std::vector<std::shared_ptr<qasm3::Statement>> program;
    // Parse the program
    try {
      program = parser.parseProgram();
      std::cout << "Parsing successful!" << std::endl;

      // You can now process the AST (program) as needed
      // For example, print out the statements
      for (const auto& statement : program) {
        std::cout << statement << std::endl;
      }
    } catch (const std::runtime_error& e) {
        std::cerr << "Parsing failed: " << e.what() << std::endl;
        return;
    }
    // Traverse the AST
    for (const auto& statement : program) {
        if (auto gateDecl = std::dynamic_pointer_cast<qasm3::GateDeclaration>(statement)) {
            std::cout << "Gate Declaration: " << gateDecl->identifier << std::endl;
            std::cout << "  Parameters: ";
            for (const auto& param : gateDecl->parameters->identifiers) {
                std::cout << param->identifier << " ";
            }
            std::cout << std::endl;
            std::cout << "  Qubits: ";
            for (const auto& qubit : gateDecl->qubits->identifiers) {
                std::cout << qubit->identifier << " ";
            }
            std::cout << std::endl;
        } else if (auto assignStmt = std::dynamic_pointer_cast<qasm3::AssignmentStatement>(statement)) {
            std::cout << "Assignment Statement: " << assignStmt->identifier->identifier << std::endl;
        } else if (auto ifStmt = std::dynamic_pointer_cast<qasm3::IfStatement>(statement)) {
            std::cout << "If Statement" << std::endl;
        } else if (auto versionDecl = std::dynamic_pointer_cast<qasm3::VersionDeclaration>(statement)) {
            std::cout << "Version Declaration: OpenQASM " << versionDecl->version << std::endl;
        }
        // Add more cases for other statement types as needed
    }
  }
private:
  //llvm::raw_string_ostream &outputStream; // Store the tikz circuit
  std::istringstream &qasmStream;
};

} // namespace

std::unique_ptr<mlir::Pass> mqss::opt::createQASM3ToQuakePass(std::istringstream &qasmStream){
  return std::make_unique<QASM3ToQuakePass>(qasmStream);
}
