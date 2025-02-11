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
  date   February 2025
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
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Support/Plugin.h"
#include "llvm/Support/Casting.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <iomanip>
#include <regex>
#include "Passes/CodeGen.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "ir/parsers/qasm3_parser/Parser.hpp"
#include "ir/parsers/qasm3_parser/Statement.hpp"
#include "ir/parsers/qasm3_parser/Types.hpp"

using namespace mlir;
using namespace mqss::support::quakeDialect;
using IDQASMMLIR = std::map<std::string, std::map<int, int>>;

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

  void insertQASMGateIntoQuakeModule(std::string gateId,
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
      {"gphase", [&](){ throw std::runtime_error("Global phase operation is not supported!"); }},
      {"xx_minus_yy", [&](){ throw std::runtime_error("xx_minus_yy phase operation is not supported!"); }},
      {"xx_plus_yy", [&](){ throw std::runtime_error("xx_plus_yy phase operation is not supported!"); }},
      {"U", [&](){ throw std::runtime_error("Global phase operation is not supported!"); }},
      {"x", [&](){
        if (params.size() !=0 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed x gate");
        builder.create<quake::XOp>(loc, adj, params, controls, targets); }},
      {"y", [&](){
        if (params.size() !=0 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed y gate");
        builder.create<quake::YOp>(loc, adj, params, controls, targets); }},
      {"z", [&](){
        if (params.size() !=0 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed z gate");
        builder.create<quake::ZOp>(loc, adj, params, controls, targets); }},
      {"h", [&](){
        if (params.size() !=0 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed h gate");
        builder.create<quake::HOp>(loc, adj, params, controls, targets); }},
      {"ch", [&](){
        if (params.size() !=0 || controls.size()!= 1 || targets.size()!=1)
          throw std::runtime_error("ill-formed ch gate");
        builder.create<quake::HOp>(loc, adj, params, controls, targets); }},
      {"s", [&](){
        if (params.size() !=0 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed s gate");
        builder.create<quake::SOp>(loc, adj, params, controls, targets); }},
      {"sx", [&](){ // since sx is not supported, replace it by rx with pi/2 rotation
        if (params.size() !=0 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed sx gate");
        mlir::Value halfPi = createFloatValue(builder,loc,1.57079632679);
        builder.create<quake::RxOp>(loc, adj, halfPi, controls, targets);
      }},
      {"sxdg", [&](){ // since sx is not supported, replace it by rx with -pi/2 rotation
        if (params.size() !=0 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed sxdg gate");
        mlir::Value minHalfPi = createFloatValue(builder,loc,-1.57079632679);
        builder.create<quake::RxOp>(loc, adj, minHalfPi, controls, targets);
      }},
      {"t", [&](){
        if (params.size() !=0 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed t gate");
        builder.create<quake::TOp>(loc, adj, params, controls, targets); }},
      {"teleport", [&](){ throw std::runtime_error("Teleport operation is not supported!");  }},
      {"rx", [&](){
        if (params.size() !=1 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed rx gate");
        builder.create<quake::RxOp>(loc, adj, params, controls, targets); }},
      {"crx", [&](){
        if (params.size() !=1 || controls.size()!= 1 || targets.size()!=1)
          throw std::runtime_error("ill-formed crx gate");
        builder.create<quake::RxOp>(loc, adj, params, controls, targets); }},
      {"ry", [&](){
        if (params.size() !=1 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed ry gate");
        builder.create<quake::RyOp>(loc, adj, params, controls, targets); }},
      {"cry", [&](){
        if (params.size() !=1 || controls.size()!= 1 || targets.size()!=1)
          throw std::runtime_error("ill-formed cry gate");
        builder.create<quake::RyOp>(loc, adj, params, controls, targets); }},
      {"p", [&](){
        if (params.size() !=1 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed p gate");
        builder.create<quake::RzOp>(loc, adj, params, controls, targets); }},
      {"phase", [&](){
        if (params.size() !=1 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed phase gate");
        builder.create<quake::RzOp>(loc, adj, params, controls, targets); }},
      {"cphase", [&](){
        if (params.size() !=1 || controls.size()!= 1 || targets.size()!=1)
          throw std::runtime_error("ill-formed cphase gate");
        builder.create<quake::RzOp>(loc, adj, params, controls, targets); }},
      {"z", [&](){
        if (params.size() !=0 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed z gate");
        builder.create<quake::RzOp>(loc, adj, params, controls, targets); }},
      {"cz", [&](){
        if (params.size() !=0 || controls.size()!= 1 || targets.size()!=1)
          throw std::runtime_error("ill-formed z gate");
        builder.create<quake::RzOp>(loc, adj, params, controls, targets); }},
      {"id", [&](){ /* do nothing because identity*/ }},
      {"cx", [&](){
        if (params.size() !=0 || controls.size()!= 1 || targets.size()!=1)
          throw std::runtime_error("ill-formed cx gate");
        builder.create<quake::XOp>(loc, adj, params, controls, targets); }},
      {"CX", [&](){
        if (params.size() !=0 || controls.size()!= 1 || targets.size()!=1)
          throw std::runtime_error("ill-formed CX gate");
        builder.create<quake::XOp>(loc, adj, params, controls, targets); }},
      {"ccx", [&](){
        if (params.size() !=0 || controls.size()!= 2 || targets.size()!=1)
          throw std::runtime_error("ill-formed ccx gate");
        builder.create<quake::XOp>(loc, adj, params, controls, targets); }},
      {"cy", [&](){
        if (params.size() !=0 || controls.size()!= 1 || targets.size()!=1)
          throw std::runtime_error("ill-formed cy gate");
        builder.create<quake::YOp>(loc, adj, params, controls, targets); }},
      {"cz", [&](){
        if (params.size() !=0 || controls.size()!= 1 || targets.size()!=1)
          throw std::runtime_error("ill-formed cz gate");
        builder.create<quake::ZOp>(loc, adj, params, controls, targets); }},
      {"swap", [&](){
        if (params.size() !=0 || controls.size()!= 0 || targets.size()!=2)
          throw std::runtime_error("ill-formed swap gate");
        builder.create<quake::SwapOp>(loc, adj, params, controls, targets); }},
      {"U", [&](){ // since u is not supported, U(θ, φ, λ) = Rz(φ) * Ry(θ) * Rz(λ)
        // u2(φ, λ)
        if (params.size() != 3 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed U gate");
        builder.create<quake::RzOp>(loc, adj, params[1], controls, targets); // phi
        builder.create<quake::RyOp>(loc, adj, params[0], controls, targets); // theta
        builder.create<quake::RzOp>(loc, adj, params[2], controls, targets); // lambda
      }},
      {"u1", [&](){
        if (params.size() != 1 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed u1 gate");
        builder.create<quake::R1Op>(loc, adj, params, controls, targets); }},
      {"u2", [&](){ // since u2 is not supported, it has to be decomposed
        // u2(φ, λ)
        if (params.size() != 2 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed u2 gate");
        builder.create<quake::RzOp>(loc, adj, params[0], controls, targets); // phi
        mlir::Value halfPi = createFloatValue(builder,loc,1.57079632679);
        builder.create<quake::RxOp>(loc, adj, halfPi, controls, targets); // pi/2
        builder.create<quake::RzOp>(loc, adj, params[0], controls, targets); // phi
      }},
      {"u3", [&](){
        if (params.size() != 3 || controls.size()!= 0 || targets.size()!=1)
          throw std::runtime_error("ill-formed u3 gate");
        builder.create<quake::U3Op>(loc, adj, params, controls, targets); }},
      {"iswap", [&](){ // since iswap is not supported, it has to be decomposed
        /*gate iswap q1, q2 {
          h q2;
          cx q1, q2;
          h q2;
        }*/
        if (params.size() != 0 || controls.size()!= 0 || targets.size()!=2)
          throw std::runtime_error("ill-formed iswap gate");
        builder.create<quake::HOp>(loc, adj, params, controls, targets[1]); // q2
        builder.create<quake::XOp>(loc, adj, params, targets[0], targets[1]); // q1, q2
        builder.create<quake::HOp>(loc, adj, params, controls, targets[1]); // q2
      }},
      {"iswapdg", [&](){ // since iswapdg is not supported, it has to be decomposed
        /*gate iswapdg q1, q2 {
            h q2;
            cx q1, q2;
            h q2;
            cz q1, q2;
            h q2;
        }*/
        if (params.size() != 0 || controls.size()!= 0 || targets.size()!=2)
          throw std::runtime_error("ill-formed iswapdg gate");
        builder.create<quake::HOp>(loc, adj, params, controls, targets[1]); // q2
        builder.create<quake::XOp>(loc, adj, params, targets[0], targets[1]); // q1, q2
        builder.create<quake::HOp>(loc, adj, params, controls, targets[1]); // q2
        builder.create<quake::ZOp>(loc, adj, params, targets[0], targets[1]); // q1, q2
        builder.create<quake::HOp>(loc, adj, params, controls, targets[1]); // q2
      }},
      {"rxx", [&](){ // since rxx is not supported, it has to be decomposed
        /*gate rxx(theta) a, b {
            cx a, b;
            rx(theta) b;
            cx a, b;
        }*/
        if (params.size() !=1 || controls.size()!= 0 || targets.size()!=2)
          throw std::runtime_error("ill-formed rxx gate");
        builder.create<quake::XOp>(loc, adj, targets[0], targets[1]);// cx a, b;
        builder.create<quake::RxOp>(loc, adj,params,controls,targets[1]);// rx b;
        builder.create<quake::XOp>(loc, adj, targets[0], targets[1]);// cx a, b;
      }},
      {"ryy", [&](){ // since ryy is not supported, it has to be decomposed
        /*gate ryy(theta) a, b {
            ry(pi/2) a;
            ry(pi/2) b;
            cx a, b;
            ry(theta) b;
            cx a, b;
            ry(-pi/2) a;
            ry(-pi/2) b;
        }*/
        if (params.size() !=1 || controls.size()!= 0 || targets.size()!=2)
          throw std::runtime_error("ill-formed ryy gate");
        mlir::Value halfPi = createFloatValue(builder,loc,1.57079632679);
        mlir::Value minHalfPi = createFloatValue(builder,loc,-1.57079632679);
        builder.create<quake::RyOp>(loc,adj,halfPi,controls,targets[0]);//ry(pi/2) a;
        builder.create<quake::RyOp>(loc,adj,halfPi,controls,targets[1]);//ry(pi/2) b;
        builder.create<quake::XOp>(loc, adj, targets[0], targets[1]);// cx a, b;
        builder.create<quake::RyOp>(loc,adj,params,controls,targets[1]);//ry(theta) b;
        builder.create<quake::XOp>(loc, adj, targets[0], targets[1]);// cx a, b;
        builder.create<quake::RyOp>(loc,adj,minHalfPi,controls,targets[0]);//ry(pi/2) a;
        builder.create<quake::RyOp>(loc,adj,minHalfPi,controls,targets[1]);//ry(pi/2) b;
      }},
      {"rzz", [&](){ // since rzz is not supported, it has to be decomposed
        /*gate rzz(theta) a, b {
            cx a, b;
            rz(theta) b;
            cx a, b;
        }*/
        if (params.size() !=1 || controls.size()!= 0 || targets.size()!=2)
          throw std::runtime_error("ill-formed rzz gate");
        builder.create<quake::XOp>(loc, adj, targets[0], targets[1]);// cx a, b;
        builder.create<quake::RzOp>(loc, adj,params,controls,targets[1]);//rz(theta) b
        builder.create<quake::XOp>(loc, adj, targets[0], targets[1]);// cx a, b;
      }},
      {"rzx", [&](){ // since rzx is not supported, it has to be decomposed
        /*gate rzx(theta) a, b {
            h b;
            cx a, b;
            rz(theta) b;
            cx a, b;
            h b;
        }*/
        if (params.size() !=1 || controls.size()!= 0 || targets.size()!=2)
          throw std::runtime_error("ill-formed rzx gate");
        builder.create<quake::HOp>(loc, adj, controls, targets[1]);// h b;
        builder.create<quake::XOp>(loc, adj, targets[0], targets[1]);// cx a, b;
        builder.create<quake::RzOp>(loc, adj,params,controls,targets[1]);//rz(theta) b
        builder.create<quake::XOp>(loc, adj, targets[0], targets[1]);// cx a, b;
        builder.create<quake::HOp>(loc, adj, controls, targets[1]);// h b;
      }},
      {"dcx", [&](){ // since dcx is not supported, it has to be decomposed
        /*gate dcx a, b {
            cx a, b;
            cx b, a;
        }*/
        if (params.size() !=0 || controls.size()!= 0 || targets.size()!=2)
          throw std::runtime_error("ill-formed dcx gate");
        builder.create<quake::XOp>(loc, adj, controls, targets[0], targets[1]);// cx a, b;
        builder.create<quake::XOp>(loc, adj, controls, targets[1], targets[0]);// cx b, a;
      }},
      {"ecr", [&](){ // since ecr is not supported, it has to be decomposed
        /*gate ecr a, b {
            h b;
            cx a, b;
            rz(pi/2) b;
            cx a, b;
            h b;
        }*/
        if (params.size() !=0 || controls.size()!= 0 || targets.size()!=2)
          throw std::runtime_error("ill-formed ecr gate");
        builder.create<quake::HOp>(loc, adj, controls, targets[1]);// h b;
        builder.create<quake::XOp>(loc, adj, controls, targets[0], targets[1]);// cx a, b;
        mlir::Value halfPi = createFloatValue(builder,loc,1.57079632679);
        builder.create<quake::RzOp>(loc, adj,halfPi, controls, targets[1]);// rz(pi/2) b;
        builder.create<quake::XOp>(loc, adj, controls, targets[0], targets[1]);// cx a, b;
        builder.create<quake::HOp>(loc, adj, controls, targets[1]);// h b;
      }},
      {"cswap", [&](){
        if (params.size() !=0 || controls.size()!= 1 || targets.size()!=2)
          throw std::runtime_error("ill-formed cswap gate");
        builder.create<quake::SwapOp>(loc, adj, params, controls, targets);}}
    };
    auto it = gateMap.find(gateId);
    if (it != gateMap.end()) {
        it->second();  // Execute the corresponding gate creation function
    } else {
        throw std::runtime_error( "Unknown gate: "+gateId+"\n");
    }
  }

  // returns a tuple of T1 and T2
  // T1 a map of mlir indices: identifier, qasmqubit, mllirqubit
  // T2 allocated vector of qubits, all the qubit registers are condensed into one
  std::tuple<IDQASMMLIR, mlir::Value>
    insertAllocatedQubits(const std::vector<
                          std::shared_ptr<qasm3::Statement>>& program,
                          mlir::func::FuncOp circuit,
                          mlir::Operation *inOp ) {
    std::map<std::string, int> expQubits;
    int totalQubits = 0;
    for (const auto& statement : program) {
      // Check if the statement is a DeclarationStatement
      if (auto declStmt = std::dynamic_pointer_cast<
                             qasm3::DeclarationStatement>(statement)) {
//        std::cout << "type name " <<  typeid(*statement).name()  << "\n";
        //std::cout << "identifier " <<  declStmt->identifier  << "\n";
        //std::cout << "expression " <<  declStmt->expression << "\n";
        // Checking the type contained in the variant
        auto& variantType = declStmt->type; 
        if (auto designatedPtr = std::get_if<std::shared_ptr<
                                  qasm3::Type<std::shared_ptr<
                                    qasm3::Expression>>>>(&variantType)) {
          // If we successfully got the Type<std::shared_ptr<Expression>>, handle it
          std::shared_ptr<qasm3::Type<std::shared_ptr<qasm3::Expression>>>
                                                    typeExpr = *designatedPtr;
          //std::cout << "Type expression to string " << typeExpr->toString() << "\n";
          //std::cout << "Successfully cast to Type<std::shared_ptr<Expression>>!" << std::endl;
          std::regex pattern("qubit"); // Case-sensitive regex
          if (!std::regex_search(typeExpr->toString(), pattern)) continue; // error code
          if (auto designator = typeExpr->getDesignator()) {
            if (auto constant = std::dynamic_pointer_cast<
                                      qasm3::Constant>(designator)) {
              expQubits.insert({std::string(declStmt->identifier),constant->getSInt()});
              totalQubits += constant->getSInt();  // Access the variant
              //std::cout << "Total qubits: " << val << std::endl;
            }
          }
        }
      }
    }
    // identifier, qasmqubit, mlirqubit
    IDQASMMLIR mlirQubits;
    if (totalQubits == 0 || totalQubits == -1)
      return std::make_tuple(mlirQubits,nullptr); // do nothing
    // instead of returning do the insertion of the measurement in the MLIR module
    //return totalQubits;
    OpBuilder builder(circuit.getContext());
    Location loc = circuit.getLoc();
    builder.setInsertionPoint(inOp);  // Set insertion before return
    // Define the type for a vector of totalQubits qubits
    auto qubitVecType = quake::VeqType::get(builder.getContext(), totalQubits);
    // Create the quake.alloca operation
    auto qubitReg = builder.create<quake::AllocaOp>(loc, qubitVecType);
    int globalCount  = 0;
    for(const auto& pair : expQubits){
      std::map<int, int> innerMap;
      for(int i = 0; i<pair.second; i++){
        innerMap.insert({i,globalCount++});
      }
      mlirQubits.insert({pair.first, innerMap});
    }
    if (globalCount != totalQubits) // paranoia checks
      throw std::runtime_error("Fatal error! This should never happen");
    return std::make_tuple(mlirQubits, qubitReg.getResult());
  }

  // Function to print gate information
  void insertGate(const std::shared_ptr<qasm3::GateCallStatement>& gateCall,
                  mlir::func::FuncOp circuit,
                  mlir::Operation *inOp,
                  mlir::Value qubits,
                  IDQASMMLIR mlirQubits) {
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
        mlir::Value argMlirVal = createFloatValue(builder,loc,argVal);
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
        if (qubitOp == -1) throw std::runtime_error("Fatal error, this must not happen!");
        int selectedQubit = -1;
        try{
          selectedQubit = mlirQubits.at(std::string(operand->identifier)).at(qubitOp);
        } catch(const std::out_of_range& e){
          throw std::runtime_error("Fatal error!");
        }                                                  //if (operand->expression) {
        //  std::cout << "[" << qubitOp << "]";
        //}
        if (i < numControls) {
          auto controlQubit = builder.create<quake::ExtractRefOp>(loc,
                                                                  qubits,
                                                                  selectedQubit);
          controls.push_back(controlQubit);
        } else {
          auto targetQubit = builder.create<quake::ExtractRefOp>(loc,
                                                                 qubits,
                                                                 selectedQubit);
          targets.push_back(targetQubit);
          //std::cout << " (Target)";
        }
        //std::cout << std::endl;
      }
    }
    std::regex pattern("dg"); // Case-sensitive regex
    if (std::regex_search(std::string(gateCall->identifier), pattern)) isAdj = true;

    insertQASMGateIntoQuakeModule(std::string(gateCall->identifier),
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
                           mlir::Value qubits,
                           IDQASMMLIR mlirQubits) {
    for (const auto& statement : program)
    // Check if the statement is a GateCallStatement
      if (auto gateCall = std::dynamic_pointer_cast<
                              qasm3::GateCallStatement>(statement))
        insertGate(gateCall, circuit, inOp, qubits, mlirQubits);
  }

  // Function to print the source qubit and target bit of each measurement
  void parseAndInsertMeasurements(const std::vector<std::shared_ptr<qasm3::Statement>>&
                                  statements,
                                  mlir::func::FuncOp circuit,
                                  mlir::Operation *inOp,
                                  mlir::Value allocatedQubits,
                                  IDQASMMLIR mlirQubits) {
    // Defining the builder
    OpBuilder builder(circuit.getContext());
    Location loc = circuit.getLoc();
    builder.setInsertionPoint(inOp);  // Set insertion before return
    //llvm::outs() << "Printing measurements!\n";
    for (const auto& statement : statements) {
      //llvm::outs() << "Statement Type: " << typeid(*statement).name() << "\n";
      // Check if the statement is a MeasureStatement
      if (auto assignmentStmt = std::dynamic_pointer_cast<
                                  qasm3::AssignmentStatement>(statement)) {
        //llvm::outs() << "Found AssignmentStatement\n";
        if (assignmentStmt->expression) {
          //llvm::outs() << "Expression Type: " << typeid(assignmentStmt->expression).name() << "\n";
          // Check if it's a DeclarationExpression (which holds the initializer)
          if (auto declExpr = std::dynamic_pointer_cast<
                              qasm3::DeclarationExpression>(assignmentStmt->expression)) {
            //llvm::outs() << "Found DeclarationExpression\n";
            // Check if the initializer is a MeasureExpression
            if (auto measureExpr = std::dynamic_pointer_cast<
                                qasm3::MeasureExpression>(declExpr->expression)) {
              //llvm::outs() << "Found MeasureExpression\n";
              if (measureExpr->gate) {
                std::string qubit = measureExpr->gate->identifier;
                //llvm::outs() << "Measured Qubit: " << qubit << "\n";
                if (measureExpr->gate->expression) {
                  //llvm::outs() << "Has expresion\n";
                  if (auto operand = std::dynamic_pointer_cast<
                              qasm3::Constant>(measureExpr->gate->expression)) {
                    size_t localQubit = operand->getSInt();
                    size_t measQubit = -1;
                    try{
                      measQubit = mlirQubits.at(qubit).at(localQubit);
                    } catch(const std::out_of_range& e){
                      throw std::runtime_error("Fatal error!");
                    }
                    // insert measurement
                    auto measRef = builder.create<quake::ExtractRefOp>(loc, allocatedQubits, measQubit);
                    SmallVector<Value> targetValues = {measRef};
                    Type measTy = quake::MeasureType::get(builder.getContext());
                    builder.create<quake::MzOp>(loc, measTy, targetValues).getMeasOut();
                    //llvm::outs() << "Operand Identifier: " << operand->getSInt() << "\n";
                  }
                }
              } else {
                throw std::runtime_error("Measurement has not qubit associated to it!");
              }
            }
          }
        }
      }
    }
  }

  class QASM3ToQuakePass
      : public PassWrapper<QASM3ToQuakePass, OperationPass<func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QASM3ToQuakePass)
  
    QASM3ToQuakePass(std::istringstream &qasmStream,
                     bool measureAllQubits) : qasmStream(qasmStream),
                                              measureAllQubits(measureAllQubits) {}
  
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
      throw std::runtime_error("Error!");
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
    auto [mlirQubits, allocatedQubits] = insertAllocatedQubits(program,circuit,returnOp);
    if(!allocatedQubits) return;  // if no allocated qubits return nothing
    // for debuggin print maps of qubits
    #ifdef DEBUG
      for(const auto& pair : mlirQubits){
        std::map<int, int> innerMap = pair.second;
        for(const auto& innerPair : innerMap){
          llvm::outs() << "id:" << pair.first << " qasmQubit: "<<innerPair.first << " mlirQubit: " << innerPair.second << "\n";
        }
      }
    #endif
    // Parse and print gate information
    parseAndInsertGates(program, circuit, returnOp, allocatedQubits, mlirQubits);
    //// Insert barriers if required
    if (!measureAllQubits)
      parseAndInsertMeasurements(program,circuit, returnOp, allocatedQubits, mlirQubits);
    else{
      // apply measurements in all allocated qubits
      // Defining the builder
      OpBuilder builder(circuit.getContext());
      Location loc = circuit.getLoc();
      builder.setInsertionPoint(returnOp);  // Set insertion before return
      Type measTy = quake::MeasureType::get(builder.getContext());
      auto stdVectType = cudaq::cc::StdvecType::get(measTy);
      builder.create<quake::MzOp>(loc, stdVectType, allocatedQubits);//  .getMeasOut();
    }
  }
private:
  std::istringstream &qasmStream;
  bool measureAllQubits=false;
};

} // namespace

std::unique_ptr<mlir::Pass> mqss::opt::createQASM3ToQuakePass(std::istringstream &qasmStream, bool measureAllQubits){
  return std::make_unique<QASM3ToQuakePass>(qasmStream, measureAllQubits);
}
