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

  // Function to determine if a gate is a multi-qubit gate with implicit controls
  bool isMultiQubitGate(const std::string& gateType) {
    return gateType == "cx" ||
           gateType == "cy" ||
           gateType == "cz" ||
           gateType == "ch" ||
           gateType == "ccx" ||
           gateType == "cswap";
  }

  // Function to get the number of controls for a gate
  size_t getNumControls(const std::string& gateType) {
    if (gateType == "cx") return 1;       // CNOT gate has 1 control
    if (gateType == "cy") return 1;       // Cy gate has 1 control
    if (gateType == "cz") return 1;       // Cz gate has 1 control
    if (gateType == "ch") return 1;       // Ch gate has 1 control
    if (gateType == "ccx") return 2;      // Toffoli gate has 2 controls
    if (gateType == "cswap") return 1;    // CSWAP gate has 1 control
    return 0;                             // Single-qubit gates have no controls
  }

  void insertGateIntoQuakeModule(std::string gateId,
                                 OpBuilder builder,
                                 Location loc,
                                 std::vector<mlir::Value> vecParams,
                                 std::vector<mlir::Value> vecControls,
                                 std::vector<mlir::Value> vecTargets,
                                 bool adj){
    mlir::ValueRange params(vecParams);
    mlir::ValueRange controls(vecControls);
    mlir::ValueRange targets(vecTargets);
    static const std::unordered_map<std::string,std::function<void()>>gateMap = {
      {"x", [&](){ builder.create<quake::XOp>(loc, adj, params, controls, targets); }},
      {"y", [&](){ builder.create<quake::YOp>(loc, adj, params, controls, targets); }},
      {"z", [&](){ builder.create<quake::ZOp>(loc, adj, params, controls, targets); }},
      {"h", [&](){ builder.create<quake::HOp>(loc, adj, params, controls, targets); }},
      {"s", [&](){ builder.create<quake::SOp>(loc, adj, params, controls, targets); }},
      {"t", [&](){ builder.create<quake::TOp>(loc, adj, params, controls, targets); }},
      {"rx", [&](){ builder.create<quake::RxOp>(loc, adj, params, controls, targets); }},
      {"cx", [&](){ builder.create<quake::XOp>(loc, adj, params, controls, targets); }},
      {"cy", [&](){ builder.create<quake::YOp>(loc, adj, params, controls, targets); }},
      {"cz", [&](){ builder.create<quake::ZOp>(loc, adj, params, controls, targets); }},
      {"swap", [&](){ builder.create<quake::SwapOp>(loc, adj, params, controls, targets); }},
      {"ccx", [&](){ builder.create<quake::XOp>(loc, adj, params, controls, targets); }},
      {"cswap", [&](){ builder.create<quake::SwapOp>(loc, adj, params, controls, targets);}}
  };
    auto it = gateMap.find(gateId);
    if (it != gateMap.end()) {
        it->second();  // Execute the corresponding gate creation function
    } else {
        throw std::runtime_error( "Unknown gate: "+gateId+"\n");
    }
  }

  mlir::Value insertAllocatedQubits(const std::vector<
                                        std::shared_ptr<qasm3::Statement>>& program,
                                    mlir::func::FuncOp circuit,
                                    mlir::Operation *inOp ) {
    int totalQubits = 0;
    for (const auto& statement : program) {
      // Check if the statement is a DeclarationStatement
      if (auto declStmt = std::dynamic_pointer_cast<
                             qasm3::DeclarationStatement>(statement)) {
//        std::cout << "type name " <<  typeid(*statement).name()  << "\n";
//        std::cout << "identifier " <<  declStmt->identifier  << "\n";
//        std::cout << "expression " <<  declStmt->expression << "\n";
        // Checking the type contained in the variant
        auto& variantType = declStmt->type; 
        if (auto designatedPtr = std::get_if<std::shared_ptr<
                                  qasm3::Type<std::shared_ptr<
                                    qasm3::Expression>>>>(&variantType)) {
          // If we successfully got the Type<std::shared_ptr<Expression>>, handle it
          std::shared_ptr<qasm3::Type<std::shared_ptr<qasm3::Expression>>>
                                                    typeExpr = *designatedPtr;
//          std::cout << "Type expression to string " << typeExpr->toString() << "\n";
//          std::cout << "Successfully cast to Type<std::shared_ptr<Expression>>!" << std::endl;
          std::regex pattern("qubit"); // Case-sensitive regex
          if (!std::regex_search(typeExpr->toString(), pattern)) continue; // error code
          if (auto designator = typeExpr->getDesignator()) {
            if (auto constant = std::dynamic_pointer_cast<
                                      qasm3::Constant>(designator)) {
              totalQubits += constant->getSInt();  // Access the variant
              //std::cout << "Total qubits: " << val << std::endl;
            }
          }
        }
      }
    }
    if (totalQubits == 0 || totalQubits == -1)
      return nullptr; // do nothing
    // instead of returning do the insertion of the measurement in the MLIR module
    //return totalQubits;
    OpBuilder builder(circuit.getContext());
    Location loc = circuit.getLoc();
    builder.setInsertionPoint(inOp);  // Set insertion before return
    // Define the type for a vector of totalQubits qubits
    auto qubitVecType = quake::VeqType::get(builder.getContext(), totalQubits);
    // Create the quake.alloca operation
    auto qubitReg = builder.create<quake::AllocaOp>(loc, qubitVecType);
    return qubitReg.getResult();
  }

  // Function to print gate information
  void insertGate(const std::shared_ptr<qasm3::GateCallStatement>& gateCall,
                  mlir::func::FuncOp circuit,
                  mlir::Operation *inOp,
                  mlir::Value qubits) {
    bool isAdj = false;
    std::vector<mlir::Value> parameters = {};
    std::vector<mlir::Value> controls   = {};
    std::vector<mlir::Value> targets    = {};
    // Defining the builder
    OpBuilder builder(circuit.getContext());
    Location loc = circuit.getLoc();
    builder.setInsertionPoint(inOp);  // Set insertion before return
    // Print the gate type (identifier)
    //std::cout << "Gate Type: " << gateCall->identifier << std::endl;
    // Print parameters (arguments)
    if (!gateCall->arguments.empty()) {
      //std::cout << "Parameters: ";
      for (const auto& arg : gateCall->arguments) {
        double argVal = 0.0;
        if (auto constantExprArg = std::dynamic_pointer_cast<
                                        qasm3::Constant>(arg))
          argVal = constantExprArg->getFP();
        mlir::Value argMlirVal = mqss::utils::createFloatValue(builder,loc,argVal);
        parameters.push_back(argMlirVal);
        //std::cout << argVal << " ";
      }
      //std::cout << std::endl;
    }
    // Print operands and their types (control or target)
    if (!gateCall->operands.empty()) {
      //std::cout << "Operands: " << std::endl;
      // Determine the number of controls
      size_t numControls = 0;
      for (const auto& modifier : gateCall->modifiers) {
        if (auto ctrlMod = std::dynamic_pointer_cast<
                                qasm3::CtrlGateModifier>(modifier)) {
          if (ctrlMod->expression) {
            if (auto constantExpr = std::dynamic_pointer_cast<
                                      qasm3::Constant>(ctrlMod->expression)) {
              int numControls = constantExpr->getSInt();
              //std::cout << "numControls " << numControls << "\n";
              break;
            }
          }
        }
      }
      // If no explicit controls, check if it's a multi-qubit gate with implicit controls
      if (numControls == 0 && isMultiQubitGate(gateCall->identifier)) {
        numControls = getNumControls(gateCall->identifier);
      }
      // Iterate over operands and classify them as controls or targets
      for (size_t i = 0; i < gateCall->operands.size(); ++i) {
        const auto& operand = gateCall->operands[i];
        //std::cout << "  - " << operand->identifier;
        // get the qubit index
        int qubitOp = -1;
        if (auto constantExprOp = std::dynamic_pointer_cast<
                                      qasm3::Constant>(operand->expression))
          qubitOp = constantExprOp->getSInt();
        //if (operand->expression) {
        //  std::cout << "[" << qubitOp << "]";
        //}
        if (i < numControls) {
          auto controlQubit = builder.create<quake::ExtractRefOp>(loc,
                                                                  qubits,
                                                                  qubitOp);
          controls.push_back(controlQubit);
        } else {
          auto targetQubit = builder.create<quake::ExtractRefOp>(loc,
                                                                 qubits,
                                                                 qubitOp);
          targets.push_back(targetQubit);
          //std::cout << " (Target)";
        }
        //std::cout << std::endl;
      }
    }

    std::regex pattern("dg"); // Case-sensitive regex
    if (std::regex_search(std::string(gateCall->identifier), pattern)) isAdj = true;

    insertGateIntoQuakeModule(std::string(gateCall->identifier),
                              builder,
                              loc,
                              parameters,
                              controls,
                              targets,
                              isAdj);
    //std::cout << "-------------------------" << std::endl;
  }

  // Function to parse and print gate information
  void parseAndInsertGates(const std::vector<std::shared_ptr<qasm3::Statement>>& program,
                           mlir::func::FuncOp circuit,
                           mlir::Operation *inOp,
                           mlir::Value qubits) {
    for (const auto& statement : program)
    // Check if the statement is a GateCallStatement
      if (auto gateCall = std::dynamic_pointer_cast<
                              qasm3::GateCallStatement>(statement))
        insertGate(gateCall, circuit, inOp, qubits);
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
    mlir::Value allocatedQubits = insertAllocatedQubits(program,circuit,returnOp);
    if(!allocatedQubits) return;  // if no allocated qubits return nothing
    // Parse and print gate information
    parseAndInsertGates(program,circuit,returnOp,allocatedQubits);
    // Parse measurements
  }
private:
  std::istringstream &qasmStream;
};

} // namespace

std::unique_ptr<mlir::Pass> mqss::opt::createQASM3ToQuakePass(std::istringstream &qasmStream){
  return std::make_unique<QASM3ToQuakePass>(qasmStream);
}
