/**
 * @file QirFunctionValidation.hpp
 * @brief Declaration of the 'QirFunctionValidationPass' class.
 */

#pragma once

#include "llvm.hpp"

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm {

/**
 * @struct FunctionValidation
 * @brief TODO
 */
struct FunctionValidation {
  std::vector<PoisonValue *> poisoned_instructions; /**< TODO */
  std::vector<UndefValue *> undefined_instructions; /**< TODO */

  bool qubits_present;  /**< TODO */
  bool results_present; /**< TODO */
};

/**
 * @class QirFunctionValidationPass
 * @brief This analysis pass looks for instructions with poisoned and undefined
 * values.
 */
class QirFunctionValidationPass
    : public AnalysisInfoMixin<QirFunctionValidationPass> {
public:
  using Result = FunctionValidation;

  Result ValidationResult;

  /**
   * @brief Applies this pass to the function 'function'.
   *
   * @param module The function.
   * @param FAM The function analysis manager.
   * @return PreservedAnalyses
   */
  PreservedAnalyses run(Function &function, FunctionAnalysisManager &FAM);
};

} // namespace llvm
