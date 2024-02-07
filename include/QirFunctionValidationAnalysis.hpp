/**
 * @file QirFunctionValidationAnalysis.hpp
 * @brief Declaration of the 'QirFunctionValidationAnalysisPass' class.
 */

#pragma once

#include "llvm.hpp"

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm
{

/**
 * @struct FunctionValidation
 * @brief TODO
 */
struct FunctionValidation
{
    std::vector<PoisonValue *> poisoned_instructions; /**< TODO */
    std::vector<UndefValue *> undefined_instructions; /**< TODO */

    bool qubits_present;  /**< TODO */
    bool results_present; /**< TODO */
};

/**
 * @class QirFunctionValidationAnalysisPass
 * @brief This analysis pass looks for instructions with poisoned and undefined
 * values.
 */
class QirFunctionValidationAnalysisPass
    : public AnalysisInfoMixin<QirFunctionValidationAnalysisPass>
{
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
    PreservedAnalyses run(Function &function, FunctionAnalysisManager &FAM,
                          QDMI_Device dev);
};

} // namespace llvm
