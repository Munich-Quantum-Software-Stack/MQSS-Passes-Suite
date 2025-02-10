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
  date   December 2024
  version 1.0
  brief
  This file contains the unitary tests for each MLIR pass in the Munich Quantum
  Software Stack (MQSS).
  * Folder code has quantum kernels writen in CudaQ (cpp).
  * Folder golden contains the expected modified quantum kernel in MLIR for each
    MLIR pass.
  1. In each test, the quantum kernel in CudaQ is lowered to QUAKE MLIR.
  2. Then the pass is applied to the QUAKE MLIR kernel.
  3. The output of the pass is compared to the expected output.
  4. Succes if both expected output and the output obtained by the pass matches.

******************************************************************************/

#include <string>
#include <iostream>
// llvm includes
#include "llvm/Support/raw_ostream.h"
// mlir includes
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"  // For translateModuleToLLVMIR
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Pass/Pass.h"
// cudaq includes
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
// includes in runtime
#include "cudaq/qis/execution_manager.h"
#include "cudaq.h"
#include "common/Executor.h"
#include "common/RuntimeMLIR.h"
#include "common/Logger.h"
#include "common/ExecutionContext.h"
#include "cudaq/spin_op.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/algorithm.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"

// test includes
#include "Passes/CodeGen.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <regex>
#define CUDAQ_GEN_PREFIX_NAME "__nvqpp__mlirgen__"

std::string getEmptyQuakeKernel(const std::string kernelName, std::string functionName){
  std::string templateEmptyQuake = 
    "module attributes {"
    "  llvm.data_layout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\", "
    "  llvm.triple = \"x86_64-unknown-linux-gnu\", "
    "  quake.mangled_name_map = {__nvqpp__mlirgen__KERNELNAME = \"FUNCTIONNAME\"}"
    "} {"
    "  func.func @__nvqpp__mlirgen__KERNELNAME() attributes {\"cudaq-entrypoint\", \"cudaq-kernel\"} {"
    "   return"
    "  }"
    ""
    "  func.func @FUNCTIONNAME(%arg0: !cc.ptr<i8>) {"
    "    return"
    "  }"
    "}"
  ;
  std::regex kernelNameRegex("KERNELNAME");
  std::regex functionNameRegex("FUNCTIONNAME");
  // Replace KERNELNAME and FUNCTIONNAME with the provided kernelName and functionName
  templateEmptyQuake = std::regex_replace(templateEmptyQuake, kernelNameRegex, kernelName);
  templateEmptyQuake = std::regex_replace(templateEmptyQuake, functionNameRegex, functionName);
  return templateEmptyQuake;
}

std::tuple<mlir::ModuleOp, mlir::MLIRContext *>
  extractMLIRContext(const std::string& quakeModule){
  auto contextPtr = cudaq::initializeMLIR();
  mlir::MLIRContext &context = *contextPtr.get();

  // Get the quake representation of the kernel
  auto quakeCode = quakeModule;
  auto m_module = mlir::parseSourceString<mlir::ModuleOp>(quakeCode, &context);
  if (!m_module)
    throw std::runtime_error("Module cannot be parsed");

  return std::make_tuple(m_module.release(), contextPtr.release());
}

std::string readFileToString(const std::string &filename) {
    std::ifstream file(filename);  // Open the file
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return "";
    }
    std::ostringstream fileContents;
    fileContents << file.rdbuf();  // Read the whole file into the string stream
    return fileContents.str();  // Convert the string stream to a string
}

TEST(TestMQSSPasses, TestQASMToQuake){
  // Open the OpenQASM 3.0 file
  std::ifstream inputQASMFile("./qasm/ghz_indep_qiskit_10.qasm");
  std::string templateEmptyQuake = getEmptyQuakeKernel("ghz_indep_qiskit_10", "_ZN3ghzILm2EEclEv");
  // Read file content into a string
  std::stringstream buffer;
  buffer << inputQASMFile.rdbuf();
  #ifdef DEBUG
    std::cout << "QASM input file:\n";
    std::cout << buffer.str() << "\n";
  #endif
  // Convert to istringstream
  std::istringstream qasmStream(buffer.str());
  // creating empty mlir modulei
  auto [mlirModule, contextPtr] = extractMLIRContext(templateEmptyQuake);
  mlir::MLIRContext &context = *contextPtr;
  #ifdef DEBUG
    std::cout << "Empty mlir module:\n";
    mlirModule->dump();
  #endif
  // creating pass manager
  mlir::PassManager pm(&context);
  pm.nest<mlir::func::FuncOp>().addPass(mqss::opt::createQASM3ToQuakePass(qasmStream));
  // running the pass
  if(mlir::failed(pm.run(mlirModule)))
    std::runtime_error("The pass failed...");
  #ifdef DEBUG
    std::cout << "Parsed Circuit from QASM:\n";
    mlirModule->dump();
  #endif
  //EXPECT_EQ(goldenOutput, std::string(moduleOutput));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
