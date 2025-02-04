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
#include "llvm/Support/Casting.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <iomanip>
#include <regex>
#include "Passes.hpp"
#include "Utils.hpp"
#include "ir/parsers/qasm3_parser/Parser.hpp"
#include "ir/parsers/qasm3_parser/Statement.hpp"
#include "ir/parsers/qasm3_parser/Types.hpp"

using namespace mlir;

  namespace {
  void insertAllocatedQubits(const std::vector<std::shared_ptr<qasm3::Statement>>& program, mlir::func::FuncOp circuit, mlir::Operation *inOp ) {
    int totalQubits = 0;
    for (const auto& statement : program) {
      // Check if the statement is a DeclarationStatement
      if (auto declStmt = std::dynamic_pointer_cast<qasm3::DeclarationStatement>(statement)) {
//        std::cout << "type name " <<  typeid(*statement).name()  << "\n";
//        std::cout << "identifier " <<  declStmt->identifier  << "\n";
//        std::cout << "expression " <<  declStmt->expression << "\n";
        // Checking the type contained in the variant
        auto& variantType = declStmt->type; 
        if (auto designatedPtr = std::get_if<std::shared_ptr<qasm3::Type<std::shared_ptr<qasm3::Expression>>>>(&variantType)) {
          // If we successfully got the Type<std::shared_ptr<Expression>>, handle it
          std::shared_ptr<qasm3::Type<std::shared_ptr<qasm3::Expression>>> typeExpr = *designatedPtr;
//          std::cout << "Type expression to string " << typeExpr->toString() << "\n";
//          std::cout << "Successfully cast to Type<std::shared_ptr<Expression>>!" << std::endl;
          std::regex pattern("qubit"); // Case-sensitive regex
          if (!std::regex_search(typeExpr->toString(), pattern)) continue; // error code
          if (auto designator = typeExpr->getDesignator()) {
            if (auto constant = std::dynamic_pointer_cast<qasm3::Constant>(designator)) {
              totalQubits += constant->getSInt();  // Access the variant
              //std::cout << "Total qubits: " << val << std::endl;
            }
          }
        }
      }
    }
    if (totalQubits == 0 || totalQubits == -1)
      return; // do nothing
    // instead of returning do the insertion of the measurement in the MLIR module
    //return totalQubits;
    OpBuilder builder(circuit.getContext());
    Location loc = circuit.getLoc();
    builder.setInsertionPoint(inOp);  // Set insertion before return
    // Define the type for a vector of totalQubits qubits
    auto qubitVecType = quake::VeqType::get(builder.getContext(), totalQubits);
    // Create the quake.alloca operation
    auto qubitReg = builder.create<quake::AllocaOp>(loc, qubitVecType);   
  }

// Function to print gate information
void printGateInfo(const std::shared_ptr<qasm3::GateCallStatement>& gateCall) {
    // Print the gate type (identifier)
    std::cout << "Gate Type: " << gateCall->identifier << std::endl;

    // Print modifiers (e.g., ctrl, inv, pow)
    if (!gateCall->modifiers.empty()) {
        std::cout << "Modifiers: ";
        for (const auto& modifier : gateCall->modifiers) {
            if (auto invMod = std::dynamic_pointer_cast<qasm3::InvGateModifier>(modifier)) {
                std::cout << "inv ";
            //} else if (auto powMod = std::dynamic_pointer_cast<qasm3::PowGateModifier>(modifier)) {
            //    std::cout << "pow(" << powMod->exponent->toString() << ") ";
            } else if (auto ctrlMod = std::dynamic_pointer_cast<qasm3::CtrlGateModifier>(modifier)) {
                std::cout << "ctrl(";
                if (ctrlMod->expression) {
                    std::cout << ctrlMod->expression;
                }
                std::cout << ") ";
            }
        }
        std::cout << std::endl;
    }

    // Print parameters (arguments)
    if (!gateCall->arguments.empty()) {
        std::cout << "Parameters: ";
        for (const auto& arg : gateCall->arguments) {
            std::cout << arg << " ";
        }
        std::cout << std::endl;
    }

    // Print targets and controls (operands)
    if (!gateCall->operands.empty()) {
        std::cout << "Operands: ";
        for (const auto& operand : gateCall->operands) {
            std::cout << operand->identifier;
            if (operand->expression) {
                std::cout << "[" << operand->expression << "]";
            }
            std::cout << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "-------------------------" << std::endl;
}

// Function to parse and print gate information
void parseAndPrintGates(const std::vector<std::shared_ptr<qasm3::Statement>>& program) {
    for (const auto& statement : program) {
        // Check if the statement is a GateCallStatement
        if (auto gateCall = std::dynamic_pointer_cast<qasm3::GateCallStatement>(statement)) {
            printGateInfo(gateCall);
        }
    }
}


  void insertGateStatement(const std::vector<std::shared_ptr<qasm3::Statement>>& program, mlir::func::FuncOp circuit, mlir::Operation *inOp ) {

  }

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
    } catch (const std::runtime_error& e) {
      llvm::outs() << "Parsing failed: " << e.what() << "\n";
      return;
    }
    // First, I do need to know the place of the return operation
    // then every new inserted operation will be before the "return" statement
    mlir::Operation *returnOp;
    circuit.walk([&](mlir::Operation *op) {
      if (isa<func::ReturnOp>(op)) {  // Check if it's a return op
        returnOp = op;
        return;
      }
    });
    // Traverse the AST
    insertAllocatedQubits(program,circuit,returnOp);
    // Parse and print gate information
    parseAndPrintGates(program);
    // Extract the number of allocated qubits
    //std::cout << "Total allocated qubits: " << totalQubits << std::endl;

  }
private:
  //llvm::raw_string_ostream &outputStream; // Store the tikz circuit
  std::istringstream &qasmStream;
};

} // namespace

std::unique_ptr<mlir::Pass> mqss::opt::createQASM3ToQuakePass(std::istringstream &qasmStream){
  return std::make_unique<QASM3ToQuakePass>(qasmStream);
}
