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


TEST(TestMQSSPasses, TestPrintQuakeGatesPass){
  std::string quakeModule = readFileToString("./golden-cases/test_PrintQuakeGatesPass.qke");
  std::string goldenOutput = readFileToString("./golden-cases/test_PrintQuakeGatesPass-golden.txt");
  std::cout << "Input Quake Module " << std::endl << quakeModule << std::endl;
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

  std::cout << "Captured output from Pass:\n" << moduleOutput << std::endl;
  EXPECT_EQ(goldenOutput, std::string(moduleOutput));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
