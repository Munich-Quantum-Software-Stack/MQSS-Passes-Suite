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

#include "Interfaces/QASMToQuake.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mqss::support::quakeDialect;

void mqss::interfaces::insertQASMGateIntoQuakeModule(
    std::string gateId, OpBuilder &builder, Location loc,
    std::vector<mlir::Value> vecParams, std::vector<mlir::Value> vecControls,
    std::vector<mlir::Value> vecTargets, bool adj) {
  mlir::ValueRange params(vecParams);
  mlir::ValueRange controls(vecControls);
  mlir::ValueRange targets(vecTargets);
#ifdef DEBUG
  std::cout << "gate " << gateId << std::endl;
  std::cout << "controls size " << controls.size() << std::endl;
  std::cout << "target size " << targets.size() << std::endl;
  std::cout << "params size " << params.size() << std::endl;
#endif
  static const std::unordered_map<std::string, std::function<void()>> gateMap =
      {{"gphase",
        [&]() { assert(false && "Global phase operation is not supported!"); }},
       {"xx_minus_yy",
        [&]() {
          assert(false && "xx_minus_yy phase operation is not supported!");
        }},
       {"xx_plus_yy",
        [&]() {
          assert(false && "xx_plus_yy phase operation is not supported!");
        }},
       {"U",
        [&]() { // since u is not supported, U(θ, φ, λ) = Rz(φ) * Ry(θ) * Rz(λ)
          // u2(φ, λ)
          assert(!(params.size() != 3 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed U gate");
          builder.create<quake::RzOp>(loc, adj, params[2], controls,
                                      targets); // phi
          builder.create<quake::RyOp>(loc, adj, params[0], controls,
                                      targets); // theta
          builder.create<quake::RzOp>(loc, adj, params[1], controls,
                                      targets); // lambda
        }},
       {"x",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed x gate");
          builder.create<quake::XOp>(loc, adj, params, controls, targets);
        }},

       {"y",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed y gate");
          builder.create<quake::YOp>(loc, adj, params, controls, targets);
        }},
       {"z",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed z gate");
          builder.create<quake::ZOp>(loc, adj, params, controls, targets);
        }},
       {"h",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed h gate");
          builder.create<quake::HOp>(loc, adj, params, controls, targets);
        }},
       {"ch",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 1 ||
                   targets.size() != 1) &&
                 "ill-formed ch gate");
          builder.create<quake::HOp>(loc, adj, params, controls, targets);
        }},
       {"s",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed s gate");
          builder.create<quake::SOp>(loc, adj, params, controls, targets);
        }},
       {"sdg",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed s gate");
          builder.create<quake::SOp>(loc, true, params, controls, targets);
        }},
       {"sx",
        [&]() { // since sx is not supported, replace it by rx with pi/2
                // rotation
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed sx gate");
          mlir::Value halfPi = createFloatValue(builder, loc, PI_2);
          builder.create<quake::RxOp>(loc, adj, halfPi, controls, targets);
          // the following sequence is also valid, but we keep the one with the
          // less number of gates
          // builder.create<quake::HOp>(loc, adj, params, controls, targets);
          // builder.create<quake::SOp>(loc, adj, params, controls, targets);
          // builder.create<quake::HOp>(loc, adj, params, controls, targets);
        }},
       {"sxdg",
        [&]() { // since sx is not supported, replace it by rx with -pi/2
                // rotation
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed sxdg gate");
          mlir::Value minHalfPi = createFloatValue(builder, loc, -1 * PI_2);
          builder.create<quake::RxOp>(loc, false, minHalfPi, controls, targets);
        }},
       {"t",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed t gate");
          builder.create<quake::TOp>(loc, adj, params, controls, targets);
        }},
       {"tdg",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed t gate");
          builder.create<quake::TOp>(loc, true, params, controls, targets);
        }},
       {"teleport",
        [&]() { assert(false && "Teleport operation is not supported!"); }},
       {"rx",
        [&]() {
          assert(!(params.size() != 1 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed rx gate");
          builder.create<quake::RxOp>(loc, adj, params, controls, targets);
        }},
       {"rz",
        [&]() {
          assert(!(params.size() != 1 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed rx gate");
          builder.create<quake::RzOp>(loc, adj, params, controls, targets);
        }},
       {"crx",
        [&]() {
          // apparently the parser get two targets
          assert(!(params.size() != 1 || controls.size() != 1 ||
                   targets.size() != 1) &&
                 "ill-formed crx gate");
          builder.create<quake::RxOp>(loc, adj, params, controls, targets);
        }},
       {"ry",
        [&]() {
          assert(!(params.size() != 1 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed ry gate");
          builder.create<quake::RyOp>(loc, adj, params, controls, targets);
        }},
       {"cry",
        [&]() {
          // apparently the parser get two targets
          assert(!(params.size() != 1 || controls.size() != 1 ||
                   targets.size() != 1) &&
                 "ill-formed cry gate");
          builder.create<quake::RyOp>(loc, adj, params, controls, targets);
        }},
       {"p",
        [&]() {
          assert(!(params.size() != 1 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed p gate");
          builder.create<quake::RzOp>(loc, adj, params, controls, targets);
        }},
       {"phase",
        [&]() {
          assert(!(params.size() != 1 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed phase gate");
          builder.create<quake::RzOp>(loc, adj, params, controls, targets);
        }},
       {"cp",
        [&]() {
          assert(!(params.size() != 1 || controls.size() != 1 ||
                   targets.size() != 1) &&
                 "ill-formed crx gate");
          double angleValue =
              extractDoubleArgumentValue(params[0].getDefiningOp());
          mlir::Value halfValue =
              createFloatValue(builder, loc, angleValue / 2);
          mlir::Value minusHalfValue =
              createFloatValue(builder, loc, -1 * angleValue / 2);
          ValueRange empty;
          builder.create<quake::RzOp>(loc, adj, halfValue, empty, controls);
          builder.create<quake::XOp>(loc, adj, empty, controls, targets);
          builder.create<quake::RzOp>(loc, adj, minusHalfValue, empty, targets);
          builder.create<quake::XOp>(loc, adj, empty, controls, targets);
          builder.create<quake::RzOp>(loc, adj, halfValue, empty, targets);
        }},
       {"cphase",
        [&]() {
          assert(!(params.size() != 1 || controls.size() != 1 ||
                   targets.size() != 1) &&
                 "ill-formed crx gate");
          double angleValue =
              extractDoubleArgumentValue(params[0].getDefiningOp());
          mlir::Value halfValue =
              createFloatValue(builder, loc, angleValue / 2);
          mlir::Value minusHalfValue =
              createFloatValue(builder, loc, -1 * angleValue / 2);
          ValueRange empty;
          builder.create<quake::RzOp>(loc, adj, halfValue, empty, controls);
          builder.create<quake::XOp>(loc, adj, empty, controls, targets);
          builder.create<quake::RzOp>(loc, adj, minusHalfValue, empty, targets);
          builder.create<quake::XOp>(loc, adj, empty, controls, targets);
          builder.create<quake::RzOp>(loc, adj, halfValue, empty, targets);
        }},
       {"z",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed z gate");
          builder.create<quake::RzOp>(loc, adj, params, controls, targets);
        }},
       {"id", [&]() { /* do nothing because identity*/ }},
       {"cx",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 1 ||
                   targets.size() != 1) &&
                 "ill-formed cx gate");
          builder.create<quake::XOp>(loc, adj, params, controls, targets);
        }},
       {"CX",
        [&]() {
          // apparently the parser get two targets
          assert(!(params.size() != 0 || controls.size() != 1 ||
                   targets.size() != 1) &&
                 "ill-formed CX gate");
          builder.create<quake::XOp>(loc, adj, params, controls, targets);
        }},
       {"ccx",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 2 ||
                   targets.size() != 1) &&
                 "ill-formed ccx gate");
          builder.create<quake::XOp>(loc, adj, params, controls, targets);
        }},
       {"cy",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 1 ||
                   targets.size() != 1) &&
                 "ill-formed cy gate");
          builder.create<quake::YOp>(loc, adj, params, controls, targets);
        }},
       {"cz",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 1 ||
                   targets.size() != 1) &&
                 "ill-formed cz gate");
          builder.create<quake::ZOp>(loc, adj, params, controls, targets);
        }},
       {"swap",
        [&]() {
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 2) &&
                 "ill-formed swap gate");
          builder.create<quake::SwapOp>(loc, adj, params, controls, targets);
        }},
       {"u",
        [&]() { // since u is not supported, U(θ, φ, λ) = Rz(φ) * Ry(θ) * Rz(λ)
          // u2(φ, λ)
          assert(!(params.size() != 3 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed U gate");
          builder.create<quake::RzOp>(loc, adj, params[2], controls,
                                      targets); // phi
          builder.create<quake::RyOp>(loc, adj, params[0], controls,
                                      targets); // theta
          builder.create<quake::RzOp>(loc, adj, params[1], controls,
                                      targets); // lambda
        }},
       {"u1",
        [&]() {
          assert(!(params.size() != 1 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed u1 gate");
          builder.create<quake::R1Op>(loc, adj, params, controls, targets);
        }},
       {"cu1",
        [&]() {
          assert(!(params.size() != 1 || controls.size() != 1 ||
                   targets.size() != 1) &&
                 "ill-formed u1 gate");
          builder.create<quake::R1Op>(loc, adj, params, controls, targets);
        }},
       {"u2",
        [&]() { // since u2 is not supported, it has to be decomposed
          // u2(φ, λ)
          assert(!(params.size() != 2 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed u2 gate");
          mlir::Value halfPi = createFloatValue(builder, loc, PI_2);
          builder.create<quake::RzOp>(loc, adj, params[1], controls,
                                      targets); // phi
          builder.create<quake::RyOp>(loc, adj, halfPi, controls,
                                      targets); // theta
          builder.create<quake::RzOp>(loc, adj, params[0], controls,
                                      targets); // lambda
        }},
       {"rccx",
        [&]() {
          mlir::Value zeroValue = createFloatValue(builder, loc, 0.0);
          mlir::Value Pi = createFloatValue(builder, loc, PI);
          mlir::Value halfPi = createFloatValue(builder, loc, PI_2);
          mlir::Value qPi = createFloatValue(builder, loc, PI_4);
          mlir::Value minusQPi = createFloatValue(builder, loc, -1 * PI_4);
          // u2 (0,pi) q[2]
          builder.create<quake::RzOp>(loc, adj, Pi, controls,
                                      targets[2]); // phi
          builder.create<quake::RyOp>(loc, adj, halfPi, controls,
                                      targets[2]); // theta
          builder.create<quake::RzOp>(loc, adj, zeroValue, controls,
                                      targets[2]); // lambda
          // u1(pi/4) q[2]
          builder.create<quake::R1Op>(loc, adj, qPi, controls, targets[2]);
          // cx q[1], q[2]
          builder.create<quake::XOp>(loc, adj, params, targets[1], targets[2]);
          // u1(-pi / 4) q[2];
          builder.create<quake::R1Op>(loc, adj, minusQPi, controls, targets[2]);
          // cx q[0], q[2];
          builder.create<quake::XOp>(loc, adj, params, targets[0], targets[2]);
          // u1(pi / 4) q[2];
          builder.create<quake::R1Op>(loc, adj, qPi, controls, targets[2]);
          // cx q[1], q[2]
          builder.create<quake::XOp>(loc, adj, params, targets[1], targets[2]);
          // u1(-pi / 4) q[2];
          builder.create<quake::R1Op>(loc, adj, minusQPi, controls, targets[2]);
          // u2 (0,pi) q[2]
          builder.create<quake::RzOp>(loc, adj, Pi, controls,
                                      targets[2]); // phi
          builder.create<quake::RyOp>(loc, adj, halfPi, controls,
                                      targets[2]); // theta
          builder.create<quake::RzOp>(loc, adj, zeroValue, controls,
                                      targets[2]); // lambda
        }},
       {"u3",
        [&]() {
          assert(!(params.size() != 3 || controls.size() != 0 ||
                   targets.size() != 1) &&
                 "ill-formed u3 gate");
          builder.create<quake::U3Op>(loc, adj, params, controls, targets);
        }},
       {"cu3",
        [&]() {
          assert(!(params.size() != 3 || controls.size() != 1 ||
                   targets.size() != 1) &&
                 "ill-formed u3 gate");
          builder.create<quake::U3Op>(loc, adj, params, controls, targets);
        }},
       {"iswap", // TODO
        [&]() {  // since iswap is not supported, it has to be decomposed
          /*gate iswap q1, q2 {
            h q2;
            cx q1, q2;
            h q2;
          }*/
          assert(false && "iswap is not supported yet!");
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 2) &&
                 "ill-formed iswap gate");
          builder.create<quake::HOp>(loc, adj, params, controls,
                                     targets[1]); // q2
          builder.create<quake::XOp>(loc, adj, params, targets[0],
                                     targets[1]); // q1, q2
          builder.create<quake::HOp>(loc, adj, params, controls,
                                     targets[1]); // q2
        }},
       {"iswapdg", // TODO
        [&]() {    // since iswapdg is not supported, it has to be decomposed
          /*gate iswapdg q1, q2 {
              h q2;
              cx q1, q2;
              h q2;
              cz q1, q2;
              h q2;
          }*/
          assert(false && "iswapdg is not supported yet!");
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 2) &&
                 "ill-formed iswapdg gate");
          builder.create<quake::HOp>(loc, false, params, controls,
                                     targets[1]); // q2
          builder.create<quake::XOp>(loc, false, params, targets[0],
                                     targets[1]); // q1, q2
          builder.create<quake::HOp>(loc, false, params, controls,
                                     targets[1]); // q2
          builder.create<quake::ZOp>(loc, false, params, targets[0],
                                     targets[1]); // q1, q2
          builder.create<quake::HOp>(loc, false, params, controls,
                                     targets[1]); // q2
        }},
       {"rxx",
        [&]() { // since rxx is not supported, it has to be decomposed
          assert(!(params.size() != 1 || controls.size() != 0 ||
                   targets.size() != 2) &&
                 "ill-formed rxx gate");
          mlir::Value zeroValue = createFloatValue(builder, loc, 0);
          mlir::Value halfPi = createFloatValue(builder, loc, PI_2);
          mlir::Value pi = createFloatValue(builder, loc, PI);
          mlir::Value minusPi = createFloatValue(builder, loc, -1 * PI);
          double angleValue =
              extractDoubleArgumentValue(params[0].getDefiningOp());
          mlir::Value minusAngle =
              createFloatValue(builder, loc, -1 * angleValue);
          mlir::Value piMinusAngle =
              createFloatValue(builder, loc, PI - angleValue);
          std::vector<mlir::Value> paramsU3;
          paramsU3.push_back(halfPi);
          paramsU3.push_back(params[0]);
          paramsU3.push_back(zeroValue);
          std::vector<mlir::Value> paramsU2;
          paramsU2.push_back(minusPi);
          paramsU2.push_back(piMinusAngle);
          std::vector<mlir::Value> paramsU1;
          paramsU1.push_back(minusAngle);
          // U3 and H
          builder.create<quake::U3Op>(loc, adj, paramsU3, controls, targets[0]);
          builder.create<quake::HOp>(loc, adj, controls, targets[1]);
          // Cx
          builder.create<quake::XOp>(loc, adj, targets[0],
                                     targets[1]); // cx a, b;
          // U1
          builder.create<quake::R1Op>(loc, adj, paramsU1, controls, targets[1]);
          // Cx
          builder.create<quake::XOp>(loc, adj, targets[0],
                                     targets[1]); // cx a, b;
          // u2
          builder.create<quake::RzOp>(loc, adj, paramsU2[1], controls,
                                      targets[0]); // phi
          builder.create<quake::RyOp>(loc, adj, halfPi, controls,
                                      targets[0]); // theta
          builder.create<quake::RzOp>(loc, adj, paramsU2[0], controls,
                                      targets[0]); // lambda
          // h
          builder.create<quake::HOp>(loc, adj, controls, targets[1]);
        }},
       {"ryy",  // TODO: not working properly yet
        [&]() { // since ryy is not supported, it has to be decomposed
          /*gate ryy(theta) a, b {
              ry(pi/2) a;
              ry(pi/2) b;
              cx a, b;
              ry(theta) b;
              cx a, b;
              ry(-pi/2) a;
              ry(-pi/2) b;
          }*/
          assert(false && "ryy is not supported yet!");
          assert(!(params.size() != 1 || controls.size() != 0 ||
                   targets.size() != 2) &&
                 "ill-formed ryy gate");

          builder.create<quake::HOp>(loc, adj, controls, targets[1]);
          builder.create<quake::XOp>(loc, adj, targets[0],
                                     targets[1]); // cx a, b;
          builder.create<quake::RzOp>(loc, adj, params, controls,
                                      targets[0]); // ry(pi/2) a;
          builder.create<quake::XOp>(loc, adj, targets[0],
                                     targets[1]); // cx a, b;
          builder.create<quake::HOp>(loc, adj, controls, targets[1]);

        }},
       {"rzz",
        [&]() { // since rzz is not supported, it has to be decomposed
          /*gate rzz(theta) a, b {
              cx a, b;
              rz(theta) b;
              cx a, b;
          }*/
          assert(!(params.size() != 1 || controls.size() != 0 ||
                   targets.size() != 2) &&
                 "ill-formed rzz gate");
          builder.create<quake::XOp>(loc, adj, targets[0],
                                     targets[1]); // cx a, b;
          builder.create<quake::R1Op>(loc, adj, params, controls,
                                      targets[1]); // rz(theta) b
          builder.create<quake::XOp>(loc, adj, targets[0],
                                     targets[1]); // cx a, b;
        }},
       {"rzx",
        [&]() { // since rzx is not supported, it has to be decomposed
          /*gate rzx(theta) a, b {
              h b;
              cx a, b;
              rz(theta) b;
              cx a, b;
              h b;
          }*/
          assert(!(params.size() != 1 || controls.size() != 0 ||
                   targets.size() != 2) &&
                 "ill-formed rzx gate");
          builder.create<quake::HOp>(loc, adj, controls, targets[1]); // h b;
          builder.create<quake::XOp>(loc, adj, targets[0],
                                     targets[1]); // cx a, b;
          builder.create<quake::RzOp>(loc, adj, params, controls,
                                      targets[1]); // rz(theta) b
          builder.create<quake::XOp>(loc, adj, targets[0],
                                     targets[1]); // cx a, b;
          builder.create<quake::HOp>(loc, adj, controls, targets[1]); // h b;
        }},
       {"dcx",
        [&]() { // since dcx is not supported, it has to be decomposed
          /*gate dcx a, b {
              cx a, b;
              cx b, a;
          }*/
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 2) &&
                 "ill-formed dcx gate");
          builder.create<quake::XOp>(loc, adj, controls, targets[0],
                                     targets[1]); // cx a, b;
          builder.create<quake::XOp>(loc, adj, controls, targets[1],
                                     targets[0]); // cx b, a;
        }},
       {"ecr",
        [&]() { // since ecr is not supported, it has to be decomposed
          /*gate ecr a, b {
              h b;
              cx a, b;
              rz(pi/2) b;
              cx a, b;
              h b;
          }*/
          assert(!(params.size() != 0 || controls.size() != 0 ||
                   targets.size() != 2) &&
                 "ill-formed ecr gate");
          mlir::Value qPi = createFloatValue(builder, loc, PI_4);
          mlir::Value minusQPi = createFloatValue(builder, loc, -1 * PI_4);
          mlir::Value Pi = createFloatValue(builder, loc, PI);
          // RZX(pi/4)
          builder.create<quake::HOp>(loc, adj, controls, targets[1]); // h b;
          builder.create<quake::XOp>(loc, adj, targets[0],
                                     targets[1]); // cx a, b;
          builder.create<quake::RzOp>(loc, adj, qPi, controls,
                                      targets[1]); // rz(theta) b
          builder.create<quake::XOp>(loc, adj, targets[0],
                                     targets[1]); // cx a, b;
          builder.create<quake::HOp>(loc, adj, controls, targets[1]); // h b;
          // rx (pi)
          builder.create<quake::RxOp>(loc, adj, Pi, controls,
                                      targets[0]); // rz(theta) b
          // RZX(-pi/4)
          builder.create<quake::HOp>(loc, adj, controls, targets[1]); // h b;
          builder.create<quake::XOp>(loc, adj, targets[0],
                                     targets[1]); // cx a, b;
          builder.create<quake::RzOp>(loc, adj, minusQPi, controls,
                                      targets[1]); // rz(theta) b
          builder.create<quake::XOp>(loc, adj, targets[0],
                                     targets[1]); // cx a, b;
          builder.create<quake::HOp>(loc, adj, controls, targets[1]); // h b;
        }},
       {"cswap", [&]() {
          assert(!(params.size() != 0 || controls.size() != 1 ||
                   targets.size() != 2) &&
                 "ill-formed cswap gate");
          builder.create<quake::SwapOp>(loc, adj, params, controls, targets);
        }}};
  auto it = gateMap.find(gateId);
  if (it != gateMap.end()) {
    it->second(); // Execute the corresponding gate creation function
  } else {
    assert(false && ("Unknown gate: " + gateId + "\n").c_str());
  }
}
