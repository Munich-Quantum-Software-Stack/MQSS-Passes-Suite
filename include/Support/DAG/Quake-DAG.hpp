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
********************************************************************************
  author Martin Letras
  date   July 2025
  version 1.0
********************************************************************************/
/** @file
  @brief
  @details This header defines a set of functions that are useful to convert
  MLIR to DAG. DAG is useful to easily track the data dependencies.
  @par
  This header must be included to use the available functions to manipulate MLIR
  modules as DAGs.
*/

#pragma once

#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_map>

using namespace mlir;

struct MLIRVertex {
  std::string name;
  mlir::Operation *operation = nullptr; // quake operation
  mlir::func::FuncOp matrix;
  mlir::Value result;
  std::vector<int> targets;
  std::vector<int> controls;
  std::vector<double> arguments;
  bool isAdj = false;
  bool isQubit = false;
  bool isMeasurement = false;
};

// Define the graph type
class QuakeDAG {
public:
  QuakeDAG() = default;

  // Parses a Quake MLIR file to build the DAG
  void parse_mlir(func::FuncOp kernel);

  // Prints the DAG to the console
  void print() const;

  // Dumps the DAG to a .dot file for Graphviz/Dotty
  void dump_dot(const std::string &filename) const;

  using DAG = boost::adjacency_list<boost::vecS, boost::vecS,
                                    boost::bidirectionalS, MLIRVertex>;

  using Vertex = boost::graph_traits<DAG>::vertex_descriptor;
  using in_edge_iterator = boost::graph_traits<DAG>::in_edge_iterator;

  DAG &getGraph() { return dag; }
  const DAG &getGraph() const { return dag; }

private:
  DAG dag;
  std::unordered_map<std::string, Vertex> node_map;

  struct VertexLabelWriter {
    const DAG &g;
    VertexLabelWriter(const DAG &graph) : g(graph) {}

    template <typename Vertex>
    void operator()(std::ostream &out, const Vertex &v) const {
      out << "[label=\"" << g[v].name << "\"]";
    }
  };
  // Adds a node to the graph if it doesn't exist
  Vertex get_or_add_node(const std::string &name);
};
