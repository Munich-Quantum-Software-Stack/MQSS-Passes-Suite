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

#include <iomanip>
#include <regex>
#include <unordered_map>

using namespace mqss::support::quakeDialect;

// Function to determine if a gate is a multi-qubit gate with implicit controls
bool mqss::interfaces::isMultiQubitGate(const std::string &gateType) {
  return gateType == "cx" || gateType == "cy" || gateType == "cz" ||
         gateType == "ch" || gateType == "ccx" || gateType == "cswap" ||
         gateType == "crx" || gateType == "cry" || gateType == "cp" ||
         gateType == "cphase" || gateType == "CX" || gateType == "cu1" ||
         gateType == "cu3";
}

// Function to get the number of controls for a gate
size_t mqss::interfaces::getNumControls(const std::string &gateType) {
  if (gateType == "cx" || gateType == "CX")
    return 1; // CNOT gate has 1 control
  if (gateType == "cy")
    return 1; // Cy gate has 1 control
  if (gateType == "crx" || gateType == "cry")
    return 1; // controlled rotations have 1 control
  if (gateType == "cp" || gateType == "cphase")
    return 1; // controlled phase have 1 control
  if (gateType == "cz")
    return 1; // Cz gate has 1 control
  if (gateType == "cu1" || gateType == "cu3")
    return 1; // Cu1 gate has 1 control
  if (gateType == "ch")
    return 1; // Ch gate has 1 control
  if (gateType == "ccx")
    return 2; // Toffoli gate has 2 controls
  if (gateType == "cswap")
    return 1; // CSWAP gate has 1 control
  return 0;   // Single-qubit gates have no controls
}

// This function returns the set of quantum registers declared in a given QASM
// program.
std::tuple<QASMVectorToQuakeVector, std::vector<std::pair<std::string, int>>>
mqss::interfaces::insertAllocatedQubits(
    const std::vector<std::shared_ptr<qasm3::Statement>> &program,
    OpBuilder &builder, Location loc, mlir::Operation *inOp) {
  // I have to do it like this to preserve the order they are declared
  std::vector<std::pair<std::string, int>> orderVectors = {};
  int totalQubits = 0;
  for (const auto &statement : program) {
    // Check if the statement is a DeclarationStatement
    if (auto declStmt =
            std::dynamic_pointer_cast<qasm3::DeclarationStatement>(statement)) {
      //        std::cout << "type name " <<  typeid(*statement).name()  <<
      //        "\n";
#ifdef DEBUG
      std::cout << "identifier " << declStmt->identifier << "\n";
      std::cout << "expression " << declStmt->expression << "\n";
#endif
      // Checking the type contained in the variant
      auto &variantType = declStmt->type;
      if (auto designatedPtr = std::get_if<
              std::shared_ptr<qasm3::Type<std::shared_ptr<qasm3::Expression>>>>(
              &variantType)) {
        // If we successfully got the Type<std::shared_ptr<Expression>>, handle
        // it
        std::shared_ptr<qasm3::Type<std::shared_ptr<qasm3::Expression>>>
            typeExpr = *designatedPtr;
        // std::cout << "Type expression to string " << typeExpr->toString() <<
        // "\n"; std::cout << "Successfully cast to
        // Type<std::shared_ptr<Expression>>!" << std::endl;
        std::regex pattern("qubit"); // Case-sensitive regex
        if (!std::regex_search(typeExpr->toString(), pattern))
          continue; // error code
        if (auto designator = typeExpr->getDesignator()) {
          if (auto constant =
                  std::dynamic_pointer_cast<qasm3::Constant>(designator)) {
            orderVectors.push_back(std::make_pair(
                std::string(declStmt->identifier), constant->getSInt()));
            totalQubits += constant->getSInt(); // Access the variant
            // std::cout << "Total qubits: " << val << std::endl;
          }
        }
      }
    }
  }
  // identifier, qasmqubit, mlirqubit
  QASMVectorToQuakeVector mlirQubitVectors;
  // IDQASMMLIR mlirQubits;
  if (totalQubits == 0 || totalQubits == -1)
    return std::make_tuple(mlirQubitVectors,
                           orderVectors); // do nothing and return empty map
  // instead of returning do the insertion of the measurement in the MLIR module
  // return totalQubits;
  builder.setInsertionPoint(inOp); // Set insertion before return
  // create the different mlir vectors in the QASM program
  for (const auto &pair : orderVectors) {
#ifdef DEBUG
    std::cout << "order " << pair.first << "\n";
#endif
    // Define the type for a vector of totalQubits qubits
    auto qubitVecType = quake::VeqType::get(builder.getContext(), pair.second);
    auto qubitReg = builder.create<quake::AllocaOp>(loc, qubitVecType);
    mlirQubitVectors.emplace(pair.first, qubitReg);
  }
  return std::make_tuple(mlirQubitVectors, orderVectors);
}

double mqss::interfaces::evaluateExpression(
    const std::shared_ptr<qasm3::Expression> &expr) {
  if (auto constantExpr = std::dynamic_pointer_cast<qasm3::Constant>(expr)) {
    double val;
    if (constantExpr->isInt() || constantExpr->isSInt() ||
        constantExpr->isUInt())
      val = constantExpr->getSInt();
    else
      val = constantExpr->getFP();
    return val;
  } else if (auto unaryExpr =
                 std::dynamic_pointer_cast<qasm3::UnaryExpression>(expr)) {
    // Handle unary expressions like -pi
    double operandValue = evaluateExpression(unaryExpr->operand);
    switch (unaryExpr->op) {
    case qasm3::UnaryExpression::Op::Negate:
      return -operandValue;
    // Add other unary operations if needed
    default:
      assert(false && "Unsupported unary operation");
    }
  } else if (auto binaryExpr =
                 std::dynamic_pointer_cast<qasm3::BinaryExpression>(expr)) {
    // Handle binary expressions like pi/2
    double lhsValue = evaluateExpression(binaryExpr->lhs);
    double rhsValue = evaluateExpression(binaryExpr->rhs);
    switch (binaryExpr->op) {
    case qasm3::BinaryExpression::Op::Add:
      return lhsValue + rhsValue;
    case qasm3::BinaryExpression::Op::Subtract:
      return lhsValue - rhsValue;
    case qasm3::BinaryExpression::Op::Multiply:
      return lhsValue * rhsValue;
    case qasm3::BinaryExpression::Op::Divide:
      return lhsValue / rhsValue;
    // Add other binary operations if needed
    default:
      assert(false && "Unsupported binary operation");
    }
  } else if (auto identifierExpr =
                 std::dynamic_pointer_cast<qasm3::IdentifierExpression>(expr)) {
    // Handle identifiers like pi
    if (identifierExpr->identifier == "pi") {
      return PI; // Use the value of pi from <cmath>
    }
    assert(false &&
           ("Unsupported identifier: " + identifierExpr->identifier).c_str());
  } else {
    assert(false && "Unsupported expression type");
  }
}

// Function that inserts a QASM gate into a MLIR/Quake module
void mqss::interfaces::insertGate(
    const std::shared_ptr<qasm3::GateCallStatement> &gateCall,
    OpBuilder &builder, Location loc, mlir::Operation *inOp,
    QASMVectorToQuakeVector QASMToVectors) {
  // mlir::Value qubits, IDQASMMLIR mlirQubits) {
  bool isAdj = false;
  std::vector<mlir::Value> parameters = {};
  std::vector<mlir::Value> controls = {};
  std::vector<mlir::Value> targets = {};
  // Defining the builder
  builder.setInsertionPoint(inOp); // Set insertion before return
  // Print the gate type (identifier)
#ifdef DEBUG
  std::cout << "Gate Type: " << gateCall->identifier << std::endl;
  std::cout << "Arguments size: " << gateCall->arguments.size() << std::endl;
#endif
  // Print parameters (arguments)
  if (!gateCall->arguments.empty()) {
#ifdef DEBUG
    std::cout << "Parameters: ";
#endif
    for (const auto &arg : gateCall->arguments) {
      double argVal = evaluateExpression(arg);
      mlir::Value argMlirVal = createFloatValue(builder, loc, argVal);
      parameters.push_back(argMlirVal);
#ifdef DEBUG
      std::cout << argVal << " ";
#endif
    }
#ifdef DEBUG
    std::cout << std::endl;
#endif
  }
  // Print operands and their types (control or target)
  if (!gateCall->operands.empty()) {
#ifdef DEBUG
    std::cout << "Operands: " << std::endl;
#endif
    // Determine the number of controls
    size_t numControls = 0;
    for (const auto &modifier : gateCall->modifiers) {
      if (auto ctrlMod =
              std::dynamic_pointer_cast<qasm3::CtrlGateModifier>(modifier)) {
        if (ctrlMod->expression) {
          if (auto constantExpr = std::dynamic_pointer_cast<qasm3::Constant>(
                  ctrlMod->expression)) {
            int numControls = constantExpr->getSInt();
#ifdef DEBUG
            std::cout << "numControls " << numControls << "\n";
#endif
            break;
          }
        }
      }
    }
    // If no explicit controls, check if it's a multi-qubit gate with implicit
    // controls
    if (numControls == 0 && isMultiQubitGate(gateCall->identifier)) {
      numControls = getNumControls(gateCall->identifier);
    }
    // Iterate over operands and classify them as controls or targets
    for (size_t i = 0; i < gateCall->operands.size(); ++i) {
      const auto &operand = gateCall->operands[i];
#ifdef DEBUG
      std::cout << "  - " << operand->identifier;
#endif
      // get the qubit index
      int qubitOp = -1;
      if (auto constantExprOp =
              std::dynamic_pointer_cast<qasm3::Constant>(operand->expression))
        qubitOp = constantExprOp->getSInt();
      assert(qubitOp != -1 && "Fatal error, this must not happen!");
      mlir::Value selectedVector =
          QASMToVectors.at(std::string(operand->identifier));
      int selectedQubit = qubitOp;
#ifdef DEBUG
      if (operand->expression) {
        std::cout << "[" << qubitOp << "]";
      }
#endif
      if (i < numControls) {
        auto controlQubit = builder.create<quake::ExtractRefOp>(
            loc, selectedVector, selectedQubit);
        controls.push_back(controlQubit);
#ifdef DEBUG
        std::cout << " (Control)";
#endif
      } else {
        auto targetQubit = builder.create<quake::ExtractRefOp>(
            loc, selectedVector, selectedQubit);
        targets.push_back(targetQubit);
#ifdef DEBUG
        std::cout << " (Target)";
#endif
      }
#ifdef DEBUG
      std::cout << std::endl;
#endif
    }
  }
  std::regex pattern("dg"); // Case-sensitive regex
  if (std::regex_search(std::string(gateCall->identifier), pattern))
    isAdj = true;
  insertQASMGateIntoQuakeModule(std::string(gateCall->identifier), builder, loc,
                                parameters, controls, targets, isAdj);
#ifdef DEBUG
  std::cout << "-------------------------" << std::endl;
#endif
}

// Function that parses a given AST/QASM and inserts measurements into a
// MLIR/Quake
void mqss::interfaces::parseAndInsertMeasurements(
    const std::vector<std::shared_ptr<qasm3::Statement>> &statements,
    OpBuilder &builder, Location loc, mlir::Operation *inOp,
    QASMVectorToQuakeVector QASMToVectors) {
  std::map<int, std::pair<std::string, int>> ClassicalRegToQVector = {};
  // mlir::Value allocatedQubits, IDQASMMLIR mlirQubits) {
  // Defining the builder
  builder.setInsertionPoint(inOp); // Set insertion before return
  // llvm::outs() << "Printing measurements!\n";
  for (const auto &statement : statements) {
    // llvm::outs() << "Statement Type: " << typeid(*statement).name() << "\n";
    //   Check if the statement is a MeasureStatement
    if (auto assignmentStmt =
            std::dynamic_pointer_cast<qasm3::AssignmentStatement>(statement)) {
      // llvm::outs() << "Found AssignmentStatement\n";
      std::string classicalRegister;
      size_t classicalIndex = 0;
      if (assignmentStmt->identifier)
        classicalRegister = assignmentStmt->identifier->getName();
      if (auto idxExpr = std::dynamic_pointer_cast<qasm3::Constant>(
              assignmentStmt->indexExpression)) {
        classicalIndex = idxExpr->getSInt();
      } else {
        llvm::errs() << "Error: indexExpression is not a Constant.\n";
        assert(false && "Unsupported classical index expression");
      }
#ifdef DEBUG
      llvm::outs() << "Measurement result goes to: " << classicalRegister << "["
                   << classicalIndex << "]\n";
#endif
      if (assignmentStmt->expression) {
        // llvm::outs() << "Expression Type: " <<
        // typeid(assignmentStmt->expression).name() << "\n";
        //  Check if it's a DeclarationExpression (which holds the initializer)
        if (auto declExpr =
                std::dynamic_pointer_cast<qasm3::DeclarationExpression>(
                    assignmentStmt->expression)) {
          // llvm::outs() << "Found DeclarationExpression\n";
          //  Check if the initializer is a MeasureExpression
          if (auto measureExpr =
                  std::dynamic_pointer_cast<qasm3::MeasureExpression>(
                      declExpr->expression)) {
            if (measureExpr->gate) {
              std::string qVector = measureExpr->gate->identifier;
              // llvm::outs() << "Measured Qubit: " << qubit << "\n";
              if (measureExpr->gate->expression) {
                // llvm::outs() << "Has expression\n";
                if (auto operand = std::dynamic_pointer_cast<qasm3::Constant>(
                        measureExpr->gate->expression)) {
                  size_t localQubit = operand->getSInt();
                  ClassicalRegToQVector.emplace(
                      classicalIndex, std::make_pair(qVector, localQubit));
                  // llvm::outs() << "Operand Identifier: " <<
                  // operand->getSInt() << "\n";
                }
              }
            } else
              assert(false && "Measurement has not qubit associated to it!");
          }
        }
      }
    }
  }
  for (const auto &pair : ClassicalRegToQVector) {
    int classicalIndex = pair.first;
    auto innerPair = pair.second;
    std::string qVector = innerPair.first;
    int qubit = innerPair.second;
    mlir::Value selectedQuakeVector = QASMToVectors.at(qVector);

    // insert measurement
    auto measRef =
        builder.create<quake::ExtractRefOp>(loc, selectedQuakeVector, qubit);
    SmallVector<Value> targetValues = {measRef};
    Type measTy = quake::MeasureType::get(builder.getContext());
    builder.create<quake::MzOp>(loc, measTy, targetValues).getMeasOut();
  }
}
