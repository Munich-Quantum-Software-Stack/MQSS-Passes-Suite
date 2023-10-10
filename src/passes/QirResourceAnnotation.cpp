#include "../headers/QirResourceAnnotation.hpp"
#include "../headers/QirAllocationAnalysis.hpp"

using namespace llvm;

PreservedAnalyses QirResourceAnnotationPass::run(Module &module, ModuleAnalysisManager &MAM) {
    for (auto &function : module) {
        QirAllocationAnalysisPass QAAP;
        FunctionAnalysisManager FAM;
        QAAP.run(function, FAM);
        auto stats = QAAP.AnalysisResult;

        if (stats.usage_qubit_counts > 0) {
            std::stringstream qc{""};
            qc << stats.usage_qubit_counts;
            function.addFnAttr("num_required_qubits", qc.str());
        }

        if (stats.usage_result_counts > 0) {
            std::stringstream rc{""};
            rc << stats.usage_result_counts;
            function.addFnAttr("num_required_results", rc.str());
        }
    }

    return PreservedAnalyses::none();
}

extern "C" PassModule* createQirPass() {
    return new QirResourceAnnotationPass();
}
