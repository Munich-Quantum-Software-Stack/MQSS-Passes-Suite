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
  date   December 2024
  version 1.0
  brief
  This file contains the unitary tests for each MLIR pass in the Munich Quantum
  Software Stack (MQSS).
  * Folder code has quantum kernels written in CudaQ (cpp).
  * Folder golden contains the expected modified quantum kernel in MLIR for each
    MLIR pass.
  1. In each test, the quantum kernel in CudaQ is lowered to QUAKE MLIR.
  2. Then the pass is applied to the QUAKE MLIR kernel.
  3. The output of the pass is compared to the expected output.
  4. Success if both expected output and the output obtained by the pass
matches.

******************************************************************************/

#include <iostream>
#include <regex>
#include <string>
// llvm includes
#include "llvm/Support/raw_ostream.h"
// mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h" // For translateModuleToLLVMIR
#include "mlir/Transforms/Passes.h"
// cudaq includes
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
// includes in runtime
#include "common/RuntimeMLIR.h"
// includes mqss passes
// #include "Passes/CodeGen.hpp"
#include "Passes/Decompositions.hpp"
// #include "Passes/Examples.hpp"
#include "MQSSJobStatus.h"
#include "Passes/Transforms.hpp"
// test includes
#include "RestClient.h"

#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include <thread>

#define CUDAQ_GEN_PREFIX_NAME "__nvqpp__mlirgen__"

using ServerMessage = nlohmann::json;

std::string mqpUrl = "https://portal.quantum.lrz.de:4000/v1/";

namespace mlir {
void registerPasses() {
  // Register the passes from the TableGen-generated code
  registerMQSSOptTransformsPasses();
  registerMQSSOptDecompositionsPasses();
}
} // namespace mlir

std::tuple<mlir::ModuleOp, mlir::MLIRContext *> createEmptyMLIRModule() {
  auto contextPtr = cudaq::initializeMLIR();
  mlir::MLIRContext &context = *contextPtr.get();
  // Create an empty MLIR module
  mlir::OwningOpRef<mlir::ModuleOp> m_module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  return std::make_tuple(m_module.release(), contextPtr.release());
}

std::tuple<mlir::ModuleOp, mlir::MLIRContext *>
extractMLIRContext(const std::string &quakeModule) {
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
  std::ifstream file(filename); // Open the file
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return "";
  }
  std::ostringstream fileContents;
  fileContents << file.rdbuf(); // Read the whole file into the string stream
  return fileContents.str();    // Convert the string stream to a string
}

std::string getQuake(std::string inputFile) {
  std::string quakeModule = readFileToString(inputFile);
  return quakeModule;
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

void invokePasses(mlir::ModuleOp circuit,
                  const std::vector<std::string> &passes) {
#ifdef DEBUG
  std::cout << "Invoking Passes" << std::endl;
#endif
  // Join the vector into a single string with commas
  std::string passPipeline = llvm::join(passes, ",");
  mlir::PassManager pm(circuit.getContext());
  // Parse the pass pipeline
  llvm::StringRef passPipelineRef(passPipeline);
  std::string errMsg;
  llvm::raw_string_ostream errOs(errMsg);
  // Add additional passes if necessary
  if (failed(parsePassPipeline(passPipelineRef, pm, errOs))) {
    llvm::errs() << "Failed to parse pass pipeline: " << passPipeline << " "
                 << errOs.str() << "\n";
    return;
  }
  if (mlir::failed(pm.run(circuit)))
    std::runtime_error("The pass failed...");
}

void transpileToIQM(mlir::MLIRContext &context, mlir::ModuleOp m_module) {
  // creating pass manager
  mlir::PassManager pm(&context);
  std::string nativeGateSet[] = {"phased_rx", "z(1)"};
  cudaq::opt::BasisConversionPassOptions options;
  options.basis = nativeGateSet;
  pm.addPass(createBasisConversionPass(options));
  // pass to canonical form and remove non-used operations
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  // running the pass
  if (mlir::failed(pm.run(m_module)))
    std::runtime_error("The pass failed...");
#ifdef DEBUG
  std::cout << "Circuit after pass:\n";
  mlirModule->dump();
#endif
}

std::string lowerQuakeCodeToOpenQASM(mlir::MLIRContext &context,
                                     mlir::ModuleOp m_module) {
  std::string postCodeGenPasses = "";
  bool printIR = false;
  bool enablePassStatistics = false;
  bool enablePrintMLIREachPass = false;

  auto translation = cudaq::getTranslation("qasm2");
  std::string codeStr;
  {
    llvm::raw_string_ostream outStr(codeStr);
    m_module.getContext()->disableMultithreading();
    if (mlir::failed(translation(m_module, outStr, postCodeGenPasses, printIR,
                                 enablePrintMLIREachPass,
                                 enablePassStatistics)))
      throw std::runtime_error("Could not successfully translate to OpenQASM2");
  }
  // Regular expression to match the gate definition
  std::regex gatePattern(R"(gate\s+\S+\(param0\)\s*\{\n\})");
  // Remove the matching part from the string
  codeStr = std::regex_replace(codeStr, gatePattern, "");
  return codeStr;
}

std::string extractJobId(ServerMessage &postResponse) {
  return postResponse["uuid"].get<std::string>();
}

std::string constructGetJobPath(ServerMessage &postResponse) {
  // In order to work with the MQSS via MQP, the GetJobPath is used to get
  // the status of a job
  return mqpUrl + "job/" + extractJobId(postResponse) + "/status";
}

std::string constructGetJobPath(std::string &jobId) {
  // In order to work with the MQSS via MQP, the GetJobPath is used to get
  // the status of a job
  return mqpUrl + "job/" + jobId + "/status";
}

std::map<std::string, std::string> generateRequestHeader() {
  std::string token = "SR1PAtiOGw5RKmHIQNCD2vRDglEHFoHm2ZRmHuM7TWYkXoChii357Uk8"
                      "FQQNt1Sg"; // no-typo-check
  std::map<std::string, std::string> headers{
      {"Authorization", "Bearer " + token},
      {"Content-Type", "application/json"}};
  return headers;
}

bool jobIsDone(ServerMessage &getJobResponse) {
  auto status = getJobResponse["status"].get<std::string>();
  if (status ==
          cudaq::mqss::jobStatusToString(cudaq::mqss::JobStatus::FAILED) ||
      status ==
          cudaq::mqss::jobStatusToString(cudaq::mqss::JobStatus::CANCELLED))
    throw std::runtime_error("Job failed to execute!");

  return status ==
         cudaq::mqss::jobStatusToString(cudaq::mqss::JobStatus::COMPLETED);
}

void submitJobsToMQSS(std::string qasmCircuit, int shots) {
  RestClient client;
  nlohmann::json jobId;
  std::string jobPostPath = mqpUrl + "job";
  auto requestHeader = generateRequestHeader();
  ServerMessage j;
  // assigning circuit files object
  j["circuit"] = qasmCircuit;
  j["circuit_format"] = "qasm"; // submitting quake to mqss
  j["resource_name"] = "QExa20";
  j["shots"] = shots;
  j["no_modify"] = false;
  j["queued"] = false;
  // Post it, get the response
  jobId = client.post(jobPostPath, "", j, requestHeader);
  std::cout << "JobId " << jobId << std::endl;
  std::string jobGetPath = constructGetJobPath(jobId);
  // try to get the results
  ServerMessage statusResponse = client.get(jobGetPath, "", requestHeader);
  while (!jobIsDone(statusResponse)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 0.5 seconds
    statusResponse = client.get(jobGetPath, "", requestHeader);
  }
  auto resultResponse = client.get(
      mqpUrl + "job/" + extractJobId(jobId) + "/result", "", requestHeader);
  std::cout << "results: " << resultResponse << std::endl;
}

void applyPassesToJob(std::string circuit,
                      std::vector<std::string> listOfPasses) {
  // load mlir module and the golden output
  std::string quakeModule = getQuake(circuit);
  // #ifdef DEBUG
  std::cout << "Input Quake Module " << std::endl << quakeModule << std::endl;
  // #endif
  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  mlir::MLIRContext &context = *contextPtr;
  invokePasses(mlirModule, listOfPasses);
  // #ifdef DEBUG
  std::cout << "Circuit after pass:\n";
  mlirModule->dump();
  // #endif
  //  transpile to IQM
  // lowerQuakeCodeToOpenQASM(context, mlirModule);
  //  lower circuit to openqasm
  std::string qasmCode = lowerQuakeCodeToOpenQASM(context, mlirModule);
  // transpilation to QExa20
  // #ifdef DEBUG
  std::cout << "Lowered QASM code" << qasmCode << "\n";
  // #endif
  //  submit cicuit to MQP
  int shots = 1000;
  submitJobsToMQSS(qasmCode, shots);
}

int main(int argc, char **argv) {
  std::cout << "Launching implementation of Arslan..." << std::endl;
  // registering mqss passes
  mlir::registerPasses();
  std::vector<std::string> passes = {//"canonicalize", "cse",
                                     "CancellationDoubleCx",
                                     "HZHToX",
                                     "CommuteCxRx",
                                     "CommuteCxX",
                                     "CommuteCxZ",
                                     "CommuteRxCx",
                                     "CommuteXCx",
                                     "CommuteZCx",
                                     "HXHToZ",
                                     "HZHToX",
                                     "SwitchHZ",
                                     "NormalizeArgAngle",
                                     "CancellationNullRotation",
                                     "SwitchXH",
                                     "SwitchYH",
                                     "SwitchZH",
                                     "CxToHCzH",
                                     "CzToHCxH",
                                     "ReverseCx",
                                     "SAdjZToS",
                                     "SZToSAdj",
                                     "SwitchHX",
                                     "decomposition{enable-patterns=CXToCZ}",
                                     "decomposition{enable-patterns=SwapToCX}",
                                     "ReductionPattern"};
  applyPassesToJob("./quake/DoubleCnotCancellationPass.qke", passes);
}
