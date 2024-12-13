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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
