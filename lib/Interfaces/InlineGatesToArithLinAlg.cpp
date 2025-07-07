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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
using namespace mlir::complex;
using namespace mlir::utils;
using namespace mqss::support::quakeDialect;

using MatrixXcd =
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;

bool hasFunc(mlir::ModuleOp module, llvm::StringRef name) {
  return static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>(name));
}

mlir::func::FuncOp getFuncOp(mlir::ModuleOp module, llvm::StringRef name) {
  return module.lookupSymbol<mlir::func::FuncOp>(name);
}

/// Two‑operand multiply helper.
/// ops[0] and ops[1] must be the same element type (complex<f32|f64>)
/// and ranks among {0,1,2}. Returns a Value of the multiplied result.
Value insertMul(OpBuilder &builder, Location loc, ArrayRef<Value> ops) {
  assert(ops.size() == 2 && "insertMul requires exactly two operands");

  // Fetch the tensor type for operand 0 (we assume both have same shape/type).
  auto rt0 = ops[0].getType().dyn_cast<RankedTensorType>();
  auto rt1 = ops[1].getType().dyn_cast<RankedTensorType>();
  assert(rt0 && rt1 && "Operands must be RankedTensorType");

  auto rank0 = rt0.getRank();
  auto rank1 = rt1.getRank();
  auto eltType = rt0.getElementType().dyn_cast<ComplexType>();
  assert(eltType && eltType == rt1.getElementType().dyn_cast<ComplexType>() &&
         "Element types must match and be complex");

  // --- Case A: Scalar (rank0 == 0 && rank1 == 0) ---
  if (rank0 == 0 && rank1 == 0) {
    return builder.create<mlir::complex::MulOp>(loc, eltType, ops[0], ops[1])
        .getResult();
  }

  // --- Case B: Vector (rank0 == 1 && rank1 == 1, same shape) ---
  if (rank0 == 1 && rank1 == 1 && rt0.getShape() == rt1.getShape()) {
    // Create empty output tensor
    Value resultTensor =
        builder.create<mlir::tensor::EmptyOp>(loc, rt0.getShape(), eltType);

    auto ctx = builder.getContext();
    auto dimExpr = mlir::getAffineDimExpr(0, ctx);
    auto map =
        mlir::AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, {dimExpr});
    SmallVector<mlir::AffineMap> indexingMaps = {map, map, map};
    SmallVector<mlir::utils::IteratorType> iterTypes = {
        mlir::utils::IteratorType::parallel};

    auto genericOp = builder.create<mlir::linalg::GenericOp>(
        loc, TypeRange{rt0}, ValueRange{ops[0], ops[1]},
        ValueRange{resultTensor}, indexingMaps, iterTypes,
        /*doc=*/"",
        /*libraryCall=*/"",
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value prod = nestedBuilder.create<mlir::complex::MulOp>(
              nestedLoc, eltType, args[0], args[1]);
          nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc, prod);
        });

    return genericOp.getResult(0);
  }

  // --- Case C: Matrix × Vector → Matvec (rank0==2, rank1==1) ---
  if (rank0 == 2 && rank1 == 1 && rt0.getDimSize(1) == rt1.getDimSize(0)) {
    // Init result tensor< M x complex >
    int64_t M = rt0.getDimSize(0);
    auto resultType = RankedTensorType::get({M}, eltType);
    Value init = builder.create<mlir::tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{M}, eltType);
    return builder
        .create<linalg::MatvecOp>(loc, resultType, ValueRange{ops[0], ops[1]},
                                  ValueRange{init})
        .getResult(0);
  }

  // --- Case D: Matrix × Matrix → Matmul (rank0==2, rank1==2) ---
  if (rank0 == 2 && rank1 == 2 && rt0.getDimSize(1) == rt1.getDimSize(0)) {
    int64_t M = rt0.getDimSize(0), K = rt0.getDimSize(1), N = rt1.getDimSize(1);
    auto resultType = RankedTensorType::get({M, N}, eltType);
    Value init = builder.create<mlir::tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{M, N}, eltType);
    return builder
        .create<linalg::MatmulOp>(loc, resultType, ValueRange{ops[0], ops[1]},
                                  ValueRange{init})
        .getResult(0);
  }

  llvm_unreachable("Unsupported operand shapes for insertMul");
}

/// Insert an N‑ary multiply by folding the 2‑operand case.
/// Requires at least 2 operands.
Value insertMulN(OpBuilder &builder, Location loc, ArrayRef<Value> ops) {
  assert(ops.size() >= 2 && "Need at least two operands to multiply");

  // Start by multiplying the first two:
  SmallVector<Value, 4> two = {ops[0], ops[1]};
  Value acc = insertMul(builder, loc, two);

  // Then fold in the rest
  for (size_t i = 2, e = ops.size(); i < e; ++i) {
    two[0] = acc;
    two[1] = ops[i];
    acc = insertMul(builder, loc, two);
  }

  return acc;
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
                                               QuakeDAG &dag,
                                               OpBuilder &builder,
                                               func::FuncOp gpuFunction) {
  auto &graph = dag.getGraph();
  using DAG = boost::adjacency_list<boost::vecS, boost::vecS,
                                    boost::bidirectionalS, MLIRVertex>;
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

      if (mlirMatrix) {
        // save the matrix into the vertex
        graph[*vi].matrix = mlirMatrix;
      }
      continue;
    }
  }
}

void mqss::interfaces::insertMatricesMultiplicationsToMLIRModule(
    mlir::ModuleOp module, QuakeDAG dag, OpBuilder &builder,
    func::FuncOp gpuFunction) {
  llvm::outs() << "insert muls\n";
  mlir::Location loc = builder.getUnknownLoc();
  auto &graph = dag.getGraph();
  using DAG = boost::adjacency_list<boost::vecS, boost::vecS,
                                    boost::bidirectionalS, MLIRVertex>;
  using Vertex = boost::graph_traits<DAG>::vertex_descriptor;
  using InEdgeIterator = boost::graph_traits<DAG>::in_edge_iterator;

  auto complexType = ComplexType::get(builder.getF64Type());
  boost::graph_traits<DAG>::vertex_iterator vi, vi_end;
  // get the operation before the return, to start to insert new operations
  mlir::Block &lastBlock = gpuFunction.getBody().back();
  // Look for the return op (usually at the end)
  for (mlir::Operation &op : lastBlock) {
    if (llvm::isa<mlir::func::ReturnOp>(&op)) {
      builder.setInsertionPoint(&op); // Insert before return
      break;
    }
  }

  for (std::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
    if (graph[*vi].isQubit || graph[*vi].isMeasurement)
      continue;
    // for each vertex insert multiplications
    Vertex v = *vi;
    std::pair<InEdgeIterator, InEdgeIterator> in_edges_range =
        in_edges(v, graph);
    llvm::outs() << "Vertex to insert mul " << graph[*vi].name << "\n";
    auto fMatrix = graph[*vi].matrix;
    llvm::outs() << "\tMatrix name " << fMatrix.getName() << "\n";
    auto funcType = fMatrix.getFunctionType();
    auto resultTypes = funcType.getResults();
    // 3) Use that result type in your CallOp
    mlir::Type matrixType = resultTypes.front();
    auto matrixValue = builder
                           .create<func::CallOp>(loc, fMatrix.getName(),
                                                 matrixType, ValueRange{})
                           .getResult(0);
    // get the operands of the matrix multiplication
    SmallVector<mlir::Value> operands;
    operands.push_back(matrixValue);
    for (InEdgeIterator it = in_edges_range.first; it != in_edges_range.second;
         ++it) {
      auto edge = *it;
      auto sourceVertex = boost::source(edge, graph);
      llvm::outs() << "\tInput Vertex " << graph[sourceVertex].name << "\n";
      if (!graph[sourceVertex].result)
        continue;
      operands.push_back(graph[sourceVertex].result);
    }
    llvm::outs() << "Operands size " << operands.size() << "\n";
    if (operands.size() != 3 && operands.size() != 1) {
      Value product = insertMulN(builder, loc, operands);
      graph[*vi].result = product;
    }
  }
}
