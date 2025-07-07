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
  date   July 2025
  version 1.0
  brief
    Definition of the class used to transform mlir Quake to a DAG
representation.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#include "Support/DAG/Quake-DAG.hpp"

using namespace mqss::support::quakeDialect;

void QuakeDAG::parse_mlir(func::FuncOp kernel) {
  int numQubits = getNumberOfQubits(kernel);
  std::map<size_t, QuakeDAG::Vertex>
      qubitsHistory; // this map stores the last inserted vertex in the graph on
                     // each qubit, the key is the index of the qubit
  for (int i = 0; i < numQubits; i++) {
    Vertex qubit = get_or_add_node("q_" + std::to_string(i));
    dag[qubit].isQubit = true;
    qubitsHistory[i] = qubit;
  }
  int idx = 0;
  kernel.walk([&](Operation *op) {
    auto gate = dyn_cast<quake::OperatorInterface>(op);
    if (!gate)
      return;
    // then, the operation is a quake gate
    llvm::StringRef opName = op->getName().getStringRef();
    std::regex pattern("^quake\\.");
    std::string result = std::regex_replace(opName.str(), pattern, "");
    Vertex operation = get_or_add_node(result + "_" + std::to_string(idx++));
    std::vector<int> controls = getIndicesOfValueRange(gate.getControls());
    std::vector<int> targets = getIndicesOfValueRange(gate.getTargets());
    std::vector<double> params = getParametersValues(gate.getParameters());
    // load data into the vertex
    dag[operation].operation = op;
    dag[operation].targets = targets;
    dag[operation].controls = controls;
    dag[operation].arguments = params;
    dag[operation].isAdj = gate.isAdj();
    // insert the edges
    for (int i = 0; i < targets.size(); i++) {
      if (qubitsHistory.find(targets[i]) != qubitsHistory.end()) {
        add_edge(qubitsHistory[targets[i]], operation, dag);
        qubitsHistory[targets[i]] = operation;
      } else
        assert("This should not happen!");
    }
    for (int i = 0; i < controls.size(); i++) {
      if (qubitsHistory.find(controls[i]) != qubitsHistory.end()) {
        add_edge(qubitsHistory[controls[i]], operation, dag);
        qubitsHistory[controls[i]] = operation;
      } else
        assert("This should not happen!");
    }
  });
  // TODO read the measurements from Quake file
  // insert measurements to al qubits at the end
  Vertex mes = get_or_add_node("measurement");
  dag[mes].isMeasurement = true;
  for (const auto &[key, value] : qubitsHistory) {
    add_edge(value, mes, dag);
  }
}

// Print the DAG to console
void QuakeDAG::print() const {
  boost::graph_traits<DAG>::vertex_iterator vi, vi_end;
  for (std::tie(vi, vi_end) = boost::vertices(dag); vi != vi_end; ++vi) {
    std::cout << dag[*vi].name << " -> ";
    for (auto out : boost::make_iterator_range(adjacent_vertices(*vi, dag))) {
      std::cout << dag[out].name << ", ";
    }
    std::cout << std::endl;
  }
}

// Dump to .dot format
void QuakeDAG::dump_dot(const std::string &filename) const {
  std::ofstream ofs(filename);
  write_graphviz(ofs, dag, VertexLabelWriter(dag));
  ofs.close();
  std::cout << "DOT graph written to " << filename << std::endl;
}

// Helper to add a node if not already present
QuakeDAG::Vertex QuakeDAG::get_or_add_node(const std::string &name) {
  if (node_map.find(name) == node_map.end()) {
    Vertex v = add_vertex(dag);
    dag[v].name = name;
    node_map[name] = v;
  }
  return node_map[name];
}
