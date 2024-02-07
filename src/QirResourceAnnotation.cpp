/**
 * @file QirResourceAnnotation.cpp
 * @brief Implementation of the 'QirResourceAnnotationPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirResourceAnnotation.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from:
 * https://github.com/qir-alliance/qat/blob/main/qir/qat/Passes/StaticResourceComponent/ResourceAnnotationPass.cpp
 */

#include <QirAllocationAnalysis.hpp>
#include <QirResourceAnnotation.hpp>

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirResourceAnnotationPass::run(Module &module,
                                                 ModuleAnalysisManager &MAM,
                                                 QDMI_Device dev)
{
    for (auto &function : module)
    {
        QirAllocationAnalysisPass QAAP;
        FunctionAnalysisManager FAM;
        QAAP.run(function, FAM, dev);
        auto stats = QAAP.AnalysisResult;

        if (stats.usage_qubit_counts > 0)
        {
            std::stringstream qc{""};
            qc << stats.usage_qubit_counts;
            function.addFnAttr("num_required_qubits", qc.str());
        }

        if (stats.usage_result_counts > 0)
        {
            std::stringstream rc{""};
            rc << stats.usage_result_counts;
            function.addFnAttr("num_required_results", rc.str());
        }
    }

    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirResourceAnnotationPass' as a
 * 'PassModule'.
 * @return QirResourceAnnotationPass
 */
extern "C" PassModule *loadQirPass() { return new QirResourceAnnotationPass(); }
