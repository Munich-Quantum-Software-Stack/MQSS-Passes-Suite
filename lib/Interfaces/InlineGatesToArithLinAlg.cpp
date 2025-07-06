/* This code and any associated documentation is provided "as is"

Copyright 2025 Munich Quantum Software Stack Project

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
  date   February 2025
  version 1.0
  brief
    Definition of map used to insert quantum gates into a MLIR module. It
receives as input a tag that identifies the quantum gate, the list of arguments,
control and target qubits.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#include "Interfaces/MatricesQuantumGates.hpp"
#include "Interfaces/QuakeToLinAlg.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "Support/DAG/Quake-DAG.hpp"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <Eigen/Dense>
#include <boost/graph/graph_traits.hpp>
#include <complex>
#include <iostream>

using namespace mlir;
using namespace mqss::support::quakeDialect;

using MatrixXcd =
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;

bool hasFunc(mlir::ModuleOp module, llvm::StringRef name) {
  return static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>(name));
}

mlir::func::FuncOp getFuncOp(mlir::ModuleOp module, llvm::StringRef name) {
  return module.lookupSymbol<mlir::func::FuncOp>(name);
}

// Function: generate an MLIR constant op holding the complex matrix as
// tensor<2x2x2xf64> The last dim 2 holds real and imaginary parts interleaved
Value createComplexMatrixConstant(OpBuilder &builder, Location loc,
                                  const MatrixXcd &mat) {
  int rows = mat.rows();
  int cols = mat.cols();

  std::vector<double> data;
  data.reserve(rows * cols * 2);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      data.push_back(mat(i, j).real());
      data.push_back(mat(i, j).imag());
    }
  }

  auto tensorType =
      RankedTensorType::get({rows, cols, 2}, builder.getF64Type());
  auto denseAttr = DenseElementsAttr::get(tensorType, llvm::makeArrayRef(data));
  return builder.create<arith::ConstantOp>(loc, tensorType, denseAttr);
}

mlir::Value initializeQubit(mlir::func::FuncOp gpuFunction,
                            mlir::OpBuilder &builder) {
  mlir::Block &entryBlock = gpuFunction.getBody().front();
  builder.setInsertionPointToStart(&entryBlock);

  // this functions inserts |0⟩ State:
  auto elementType = mlir::ComplexType::get(builder.getF64Type());
  auto tensorType = mlir::RankedTensorType::get({2}, elementType);

  std::vector<std::complex<double>> data = {{1.0, 0.0}, {0.0, 0.0}};

  mlir::DenseElementsAttr initAttr =
      mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(data));
  mlir::Location loc = builder.getUnknownLoc();

  auto constantOp =
      builder.create<mlir::arith::ConstantOp>(loc, tensorType, initAttr);
  return constantOp;
}

template <typename Derived>
mlir::func::FuncOp
inlineFunction(mlir::ModuleOp &module, std::string name,
               mlir::OpBuilder &builder,
               const Eigen::MatrixBase<Derived> &matrixBase) {
  using Scalar = typename Derived::Scalar;
  static_assert(std::is_same_v<Scalar, float> ||
                    std::is_same_v<Scalar, double> ||
                    std::is_same_v<Scalar, std::complex<float>> ||
                    std::is_same_v<Scalar, std::complex<double>>,
                "Unsupported scalar type");

  if (hasFunc(module, name))
    return getFuncOp(module, name);

  builder.setInsertionPointToStart(module.getBody());
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Type floatType;
  if constexpr (std::is_same_v<Scalar, float> ||
                std::is_same_v<Scalar, std::complex<float>>) {
    floatType = builder.getF32Type();
  } else if constexpr (std::is_same_v<Scalar, double> ||
                       std::is_same_v<Scalar, std::complex<double>>) {
    floatType = builder.getF64Type();
  }

  auto complexType = mlir::ComplexType::get(floatType);

  auto shape = llvm::SmallVector<int64_t>{matrixBase.rows(), matrixBase.cols()};
  auto matrixType = mlir::RankedTensorType::get(shape, complexType);
  auto funcType = builder.getFunctionType({}, {matrixType});

  auto func = builder.create<mlir::func::FuncOp>(loc, name, funcType);
  func.setPrivate();

  mlir::Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  llvm::SmallVector<mlir::Value> elements;
  for (int i = 0; i < matrixBase.rows(); ++i) {
    for (int j = 0; j < matrixBase.cols(); ++j) {
      std::complex<double> value;

      if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                    std::is_same_v<Scalar, std::complex<double>>) {
        value = matrixBase(i, j);
      } else {
        value =
            std::complex<double>(static_cast<double>(matrixBase(i, j)), 0.0);
      }

      auto realAttr = builder.getFloatAttr(floatType, value.real());
      auto imagAttr = builder.getFloatAttr(floatType, value.imag());

      auto realConst =
          builder.create<mlir::arith::ConstantOp>(loc, floatType, realAttr);
      auto imagConst =
          builder.create<mlir::arith::ConstantOp>(loc, floatType, imagAttr);

      auto complexVal = builder.create<mlir::complex::CreateOp>(
          loc, complexType, realConst, imagConst);
      elements.push_back(complexVal);
    }
  }

  auto tensorVal =
      builder.create<mlir::tensor::FromElementsOp>(loc, matrixType, elements);
  builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{tensorVal});
  return func;
}

void mqss::interfaces::insertGatesToMLIRModule(mlir::ModuleOp module,
                                               QuakeDAG dag, OpBuilder &builder,
                                               func::FuncOp gpuFunction) {
  auto &graph = dag.getGraph();
  using DAG = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
                                    MLIRVertex>;
  boost::graph_traits<DAG>::vertex_iterator vi, vi_end;
  for (std::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
    if (graph[*vi].isQubit) {
      // insert the initial state of the qubit
      auto result = initializeQubit(gpuFunction, builder);
      graph[*vi].result = result;
      continue;
    }
    if (graph[*vi].isQubit == false && graph[*vi].isMeasurement == false) {
      // this is a gate
      std::cout << "gate name " << graph[*vi].name << std::endl;
      auto gate = graph[*vi].operation;
      gate->dump();
      QuantumGates quantumGates;
      func::FuncOp mlirMatrix;
      std::string gpuFunName = gpuFunction.getName().str();
      std::string gateName = graph[*vi].name;
      std::vector<int> controls = graph[*vi].controls;
      std::vector<int> targets = graph[*vi].targets;
      std::vector<double> arguments = graph[*vi].arguments;

      if (isa<quake::XOp>(gate)) {
        if (targets.size() == 1 && controls.size() == 0)
          mlirMatrix =
              inlineFunction(module, "Matrix_XOp", builder, quantumGates.X());
        if (targets.size() == 1 && controls.size() == 1)
          mlirMatrix = inlineFunction(module, "Matrix_CxOp", builder,
                                      quantumGates.CNOT());
      }
      if (isa<quake::YOp>(gate))
        mlirMatrix =
            inlineFunction(module, "Matrix_YOp", builder, quantumGates.Y());
      if (isa<quake::ZOp>(gate))
        mlirMatrix =
            inlineFunction(module, "Matrix_ZOp", builder, quantumGates.Z());
      if (isa<quake::HOp>(gate))
        mlirMatrix =
            inlineFunction(module, "Matrix_HOp", builder, quantumGates.H());
      if (isa<quake::SOp>(gate))
        mlirMatrix =
            inlineFunction(module, "Matrix_SOp", builder, quantumGates.S());
      if (isa<quake::TOp>(gate))
        mlirMatrix =
            inlineFunction(module, "Matrix_TOp", builder, quantumGates.T());
      if (isa<quake::R1Op>(gate)) {
        std::string matrixName = "Matrix_" + gpuFunName + "_" + gateName;
        mlirMatrix = inlineFunction(module, matrixName, builder,
                                    quantumGates.R1(arguments[0]));
      }
      if (isa<quake::RxOp>(gate)) {
        std::string matrixName = "Matrix_" + gpuFunName + "_" + gateName;
        mlirMatrix = inlineFunction(module, matrixName, builder,
                                    quantumGates.Rx(arguments[0]));
      }
      if (isa<quake::RyOp>(gate)) {
        std::string matrixName = "Matrix_" + gpuFunName + "_" + gateName;
        mlirMatrix = inlineFunction(module, matrixName, builder,
                                    quantumGates.Ry(arguments[0]));
      }
      if (isa<quake::RzOp>(gate)) {
        std::string matrixName = "Matrix_" + gpuFunName + "_" + gateName;
        mlirMatrix = inlineFunction(module, matrixName, builder,
                                    quantumGates.Rz(arguments[0]));
      }
      if (isa<quake::SwapOp>(gate))
        mlirMatrix = inlineFunction(module, "Matrix_SwapOp", builder,
                                    quantumGates.SWAP());

      if (!mlirMatrix) {
        // save the matrix into the vertex
        graph[*vi].matrix = mlirMatrix;
      }
      continue;
    }
  }
}
