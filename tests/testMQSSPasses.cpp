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

// test includes
#include "Passes.hpp"
#include <gtest/gtest.h>
#include <fstream>

#define CUDAQ_GEN_PREFIX_NAME "__nvqpp__mlirgen__"

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

std::tuple<std::string, std::string> getQuakeAndGolden(std::string cppFile,
                                                       std::string goldenFile){
  int retCode = std::system(("cudaq-quake "+cppFile+" -o ./o.qke").c_str());
  if (retCode) throw std::runtime_error("Quake transformation failed!!!");
  retCode = std::system("cudaq-opt --canonicalize --unrolling-pipeline o.qke -o ./kernel.qke");
  if (retCode) throw std::runtime_error("Quake transformation failed!!!");
  // loading the generated mlir kernel of the given cpp
  std::string quakeModule  = readFileToString("./kernel.qke");
  std::string goldenOutput = readFileToString(goldenFile);
  std::remove("./o.qke");
  std::remove("./kernel.qke");
  return std::make_tuple(quakeModule, goldenOutput);
}

std::string normalize(const std::string &str) {
    std::string result;
    for (char c : str) {
        if (c != '\t' && c != '\n' && c != '\\' && c != ' ') {
            result += c;
        }
    }
    return result;
}

TEST(TestMQSSPasses, TestPrintQuakeGatesPass){
  // load mlir module and the golden output
  auto[quakeModule, goldenOutput] =  getQuakeAndGolden(
          "./code/PrintQuakeGatesPass.cpp",
          "./golden-cases/PrintQuakeGatesPass-golden.qke");
  #ifdef DEBUG
    std::cout << "Input Quake Module " << std::endl << quakeModule << std::endl;
  #endif
  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  mlir::MLIRContext &context = *contextPtr;
  // creating pass manager
  mlir::PassManager pm(&context);
  // Adding custom pass
  std::string moduleOutput;
  llvm::raw_string_ostream stringStream(moduleOutput);
  pm.nest<mlir::func::FuncOp>().addPass(mqss::opt::createPrintQuakeGatesPass(stringStream));
  // running the pass
  if(mlir::failed(pm.run(mlirModule)))
    std::runtime_error("The pass failed...");
  #ifdef DEBUG
    std::cout << "Captured output from Pass:\n" << moduleOutput << std::endl;
  #endif
  EXPECT_EQ(goldenOutput, std::string(moduleOutput));
}

TEST(TestMQSSPasses, TestCustomExamplePass){
  // load mlir module and the golden output
  auto[quakeModule, goldenOutput] =  getQuakeAndGolden(
          "./code/CustomExamplePass.cpp",
          "./golden-cases/CustomExamplePass-golden.qke");
  #ifdef DEBUG
    std::cout << "Input Quake Module " << std::endl << quakeModule << std::endl;
  #endif
  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  mlir::MLIRContext &context = *contextPtr;
  // creating pass manager
  mlir::PassManager pm(&context);
  // Adding custom pass
  pm.addNestedPass<mlir::func::FuncOp>(mqss::opt::createCustomExamplePass());
  // running the pass
  if(mlir::failed(pm.run(mlirModule)))
    std::runtime_error("The pass failed...");
  // Convert the module to a string
  std::string moduleAsString;
  llvm::raw_string_ostream stringStream(moduleAsString);
  mlirModule->print(stringStream);

  std::cout << "Module after Pass\n" << moduleAsString << std::endl;

  EXPECT_EQ(goldenOutput, moduleAsString);
}

TEST(TestMQSSPasses, TestQuakeQMapPass01){
  // load mlir module and the golden output
  auto[quakeModule, goldenOutput] =  getQuakeAndGolden(
          "./code/QuakeQMapPass-01.cpp",
          "./golden-cases/QuakeQMapPass-01-golden.qke");
  #ifdef DEBUG
    std::cout << "Input Quake Module 01 "<<std::endl<< quakeModule << std::endl;
  #endif
  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  mlir::MLIRContext &context = *contextPtr;
  // creating pass manager
  mlir::PassManager pm(&context);
  // Defining test architecture
  Architecture arch{};
  /*
      3
     / \
    4   2
    |   |
    0---1
  */
  const CouplingMap cm = {{0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 3},
                          {3, 2}, {3, 4}, {4, 3}, {4, 0}, {0, 4}};
  arch.loadCouplingMap(5, cm);
  std::cout << "Dumping the architecture " << std::endl;
  Architecture::printCouplingMap(arch.getCouplingMap(), std::cout);
  // Defining the settings of the mqt-mapper
  Configuration settings{};
  settings.heuristic = Heuristic::GateCountMaxDistance;
  settings.layering = Layering::DisjointQubits;
  settings.initialLayout = InitialLayout::Identity;
  settings.preMappingOptimizations = false;
  settings.postMappingOptimizations = false;
  settings.lookaheadHeuristic = LookaheadHeuristic::None;
  settings.debug = false;
  settings.addMeasurementsToMappedCircuit = true;
  // Adding the QuakeQMap pass to the PassManager
  pm.nest<mlir::func::FuncOp>().addPass(mqss::opt::createQuakeQMapPass(arch,settings));
  // running the pass
  if(mlir::failed(pm.run(mlirModule)))
    std::runtime_error("The pass failed...");
  #ifdef DEBUG
    std::cout << "Mapped Circuit:\n";
    mlirModule->dump();
  #endif
  // Convert the module to a string
  std::string moduleOutput;
  llvm::raw_string_ostream stringStream(moduleOutput);
  mlirModule->print(stringStream);
  EXPECT_EQ(goldenOutput, std::string(moduleOutput));
}

TEST(TestMQSSPasses, TestQuakeQMapPass02){
  // load mlir module and the golden output
  auto[quakeModule, goldenOutput] =  getQuakeAndGolden(
          "./code/QuakeQMapPass-02.cpp",
          "./golden-cases/QuakeQMapPass-02-golden.qke");
  #ifdef DEBUG
    std::cout << "Input Quake Module 01 "<<std::endl<< quakeModule << std::endl;
  #endif
  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  mlir::MLIRContext &context = *contextPtr;
  // creating pass manager
  mlir::PassManager pm(&context);
  // Defining test architecture
  Architecture arch{};
  /*
      3
     / \
    4   2
    |   |
    0---1
  */
  const CouplingMap cm = {{0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 3},
                        {3, 2}, {3, 4}, {4, 3}, {4, 0}, {0, 4}};
  arch.loadCouplingMap(5, cm);
  std::cout << "Dumping the architecture " << std::endl;
  Architecture::printCouplingMap(arch.getCouplingMap(), std::cout);
  // Defining the settings of the mqt-mapper
  Configuration settings{};
  settings.heuristic = Heuristic::GateCountMaxDistance;
  settings.layering = Layering::DisjointQubits;
  settings.initialLayout = InitialLayout::Identity;
  settings.preMappingOptimizations = false;
  settings.postMappingOptimizations = false;
  settings.lookaheadHeuristic = LookaheadHeuristic::None;
  settings.debug = false;
  settings.addMeasurementsToMappedCircuit = true;
  // Adding the QuakeQMap pass to the PassManager
  pm.nest<mlir::func::FuncOp>().addPass(mqss::opt::createQuakeQMapPass(arch,settings));
  // running the pass
  if(mlir::failed(pm.run(mlirModule)))
    std::runtime_error("The pass failed...");
  #ifdef DEBUG
    std::cout << "Mapped Circuit:\n";
    mlirModule->dump();
  #endif
  // Convert the module to a string
  std::string moduleOutput;
  llvm::raw_string_ostream stringStream(moduleOutput);
  mlirModule->print(stringStream);
  EXPECT_EQ(goldenOutput, std::string(moduleOutput));
}

TEST(TestMQSSPasses, TestQuakeToTikzPass){
  // load mlir module and the golden output
  auto[quakeModule, goldenOutput] =  getQuakeAndGolden(
          "./code/QuakeToTikzPass.cpp",
          "./golden-cases/QuakeToTikzPass.tikz.tex");
  #ifdef DEBUG
    std::cout << "Input Quake Module " << std::endl << quakeModule << std::endl;
  #endif

  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  mlir::MLIRContext &context = *contextPtr;
  // creating pass manager
  mlir::PassManager pm(&context);
  // Adding custom pass
  std::string moduleOutput;
  llvm::raw_string_ostream stringStream(moduleOutput);
  pm.nest<mlir::func::FuncOp>().addPass(mqss::opt::createQuakeToTikzPass(stringStream));
  // running the pass
  if(mlir::failed(pm.run(mlirModule)))
    std::runtime_error("The pass failed...");
  #ifdef DEBUG
    std::cout << "Captured output from Pass:\n" << moduleOutput << std::endl;
  #endif
  EXPECT_EQ(normalize(goldenOutput), normalize(moduleOutput));
}

std::tuple<std::string,std::string> behaviouralTest(std::tuple<std::string, 
                                       std::string, 
                                       std::string, 
                                       std::function<std::unique_ptr<mlir::Pass>()>> test){
  std::string fileInputTest  = std::get<1>(test);
  std::string fileGoldenCase = std::get<2>(test);
  auto passMlir = std::get<3>(test);
  // Invoke the function to create the pass
  std::unique_ptr<mlir::Pass> pass = passMlir();

  // load mlir module and the golden output
  auto[quakeModule, goldenOutput] =  getQuakeAndGolden(
          fileInputTest,
          fileGoldenCase);
  #ifdef DEBUG
    std::cout << "Input Quake Module " << std::endl << quakeModule << std::endl;
  #endif
  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  mlir::MLIRContext &context = *contextPtr;
  // creating pass manager
  mlir::PassManager pm(&context);
  // Adding the QuakeQMap pass to the PassManager
  pm.addPass(mlir::createCanonicalizerPass());
  //auto pass = passMlir();
  pm.nest<mlir::func::FuncOp>().addPass(std::move(pass));
  // running the pass
  if(mlir::failed(pm.run(mlirModule)))
    std::runtime_error("The pass failed...");
  #ifdef DEBUG
    std::cout << "Circuit after pass:\n";
    mlirModule->dump();
  #endif
  // Convert the module to a string
  std::string moduleOutput;
  llvm::raw_string_ostream stringStream(moduleOutput);
  mlirModule->print(stringStream);
  return std::make_tuple(goldenOutput, moduleOutput);
}

class BehaviouralTestPassesMQSS : 
        public ::testing::TestWithParam<std::tuple<std::string, 
                                                   std::string, std::string, 
                                                   std::function<std::unique_ptr<mlir::Pass>()>>> {};

TEST_P(BehaviouralTestPassesMQSS , Run) {
    std::tuple<std::string, std::string, std::string, std::function<std::unique_ptr<mlir::Pass>()>> p = GetParam();
    std::string testName = std::get<0>(p);
    SCOPED_TRACE(testName);
    auto [goldenOutput, moduleOutput] = behaviouralTest(p);
    EXPECT_EQ(goldenOutput, std::string(moduleOutput));
}

INSTANTIATE_TEST_SUITE_P(
    MQSSPassTests,
    BehaviouralTestPassesMQSS,
    ::testing::Values(
      std::make_tuple("TestCxToHCzHDecompositionPass",
                      "./code/CxToHCzHDecompositionPass.cpp", 
                      "./golden-cases/CxToHCzHDecompositionPass.qke", 
                      []() { return mqss::opt::createCxToHCzHDecompositionPass();}),
      std::make_tuple("TestCzToHCxHDecompositionPass", 
                      "./code/CzToHCxHDecompositionPass.cpp",
                      "./golden-cases/CzToHCxHDecompositionPass.qke", 
                      []() { return mqss::opt::createCzToHCxHDecompositionPass();}), 
      std::make_tuple("TestCommuteCnotRxPass", 
                      "./code/CommuteCNotRxPass.cpp",
                      "./golden-cases/CommuteCNotRxPass.qke", 
                      []() { return mqss::opt::createCommuteCNotRxPass();}),
      std::make_tuple("TestCommuteCnotXPass", 
                      "./code/CommuteCNotXPass.cpp",
                      "./golden-cases/CommuteCNotXPass.qke", 
                      []() { return mqss::opt::createCommuteCNotXPass();}),
      std::make_tuple("TestCommuteCnotZPass", 
                      "./code/CommuteCNotZPass.cpp",
                      "./golden-cases/CommuteCNotZPass.qke", 
                      []() { return mqss::opt::createCommuteCNotZPass();}),
      std::make_tuple("TestCommuteRxCnotPass", 
                      "./code/CommuteRxCNotPass.cpp",
                      "./golden-cases/CommuteRxCNotPass.qke", 
                      []() { return mqss::opt::createCommuteRxCNotPass();}),
      std::make_tuple("TestCommuteXCNotPass", 
                      "./code/CommuteXCNotPass.cpp",
                      "./golden-cases/CommuteXCNotPass.qke", 
                      []() { return mqss::opt::createCommuteXCNotPass();}),
      std::make_tuple("TestCommuteZCnotPass", 
                      "./code/CommuteZCNotPass.cpp",
                      "./golden-cases/CommuteZCNotPass.qke", 
                      []() { return mqss::opt::createCommuteZCNotPass();})
    ),
  [](const ::testing::TestParamInfo<BehaviouralTestPassesMQSS::ParamType>& info) {
        // Use the first element of the tuple (testName) as the custom test name
        return std::get<0>(info.param);
  }
);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
