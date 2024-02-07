/**
 * @file QirFunctionValidationAnalysis.cpp
 * @brief Implementation of the 'QirFunctionValidationAnalysisPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirFunctionValidationAnalysis.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from:
 * https://github.com/qir-alliance/qat/blob/main/qir/qat/Passes/ValidationPass/FunctionValidationPass.cpp
 */

#include <QirAllocationAnalysis.hpp>
#include <QirFunctionValidationAnalysis.hpp>

using namespace llvm;

/**
 * @brief Applies this pass to the 'function' function.
 * @param function The function.
 * @param FAM The function analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirFunctionValidationAnalysisPass::run(
    Function &function, FunctionAnalysisManager & /*FAM*/, QDMI_Device dev)
{
    FunctionValidation result;

    QirAllocationAnalysisPass QAAP;
    FunctionAnalysisManager FAM;
    QAAP.run(function, FAM, dev);
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
