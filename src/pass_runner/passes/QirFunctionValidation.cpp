/**
 * @file QirFunctionValidation.cpp
 * @brief Implementation of the 'QirFunctionValidationPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirFunctionValidation.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from:
 * https://github.com/qir-alliance/qat/blob/main/qir/qat/Passes/ValidationPass/FunctionValidationPass.cpp
 */

#include "../headers/QirFunctionValidation.hpp"
#include "../headers/QirAllocationAnalysis.hpp"

using namespace llvm;

/**
 * @brief Applies this pass to the 'function' function.
 * @param function The function.
 * @param FAM The function analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses
QirFunctionValidationPass::run(Function &function,
                               FunctionAnalysisManager & /*FAM*/)
{
    FunctionValidation result;

    QirAllocationAnalysisPass QAAP;
    FunctionAnalysisManager FAM;
    QAAP.run(function, FAM);
    auto stats = QAAP.AnalysisResult;

    result.qubits_present = stats.usage_qubit_counts > 0 ? true : false;
    result.results_present = stats.usage_result_counts > 0 ? true : false;

    for (auto &block : function)
    {
        for (auto &instr : block)
        {
            for (auto &op : instr.operands())
            {
                auto poison = dyn_cast<PoisonValue>(op);
                auto undef = dyn_cast<UndefValue>(op);

                if (poison)
                    result.poisoned_instructions.push_back(poison);

                if (undef)
                    result.undefined_instructions.push_back(undef);
            }
        }
    }

    ValidationResult = result;

    return PreservedAnalyses::all();
}
