/**
 * @file QirAnnotateUnsupportedGates.cpp
 * @brief Implementation of the 'QirAnnotateUnsupportedGatesPass' class. <a href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirAnnotateUnsupportedGates.cpp?ref_type=heads">Go to the source code of this file.</a>
 *
 * This pass inserts an "unsupported" attribute to the appropriate gates 
 * after querying the target platform using QDMI.
 *
 * @todo libQirAnnotateUnsupportedGatesPass.so cannot be loaded even though it exists
 */

#include "../headers/QirAnnotateUnsupportedGates.hpp"

using namespace llvm;

/**
 * @var QirAnnotateUnsupportedGatesPass::QIS_START
 * @brief Used within the 'QirAnnotateUnsupportedGatesPass' to define the quantum prefix.
 */
std::string const QirAnnotateUnsupportedGatesPass::QIS_START = "__quantum"
                                                               "__qis_";

/**
 * @brief Applies this pass to the QIR's LLVM module.
 *
 * @param module The module of the submitted QIR.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirAnnotateUnsupportedGatesPass::run(Module &module, ModuleAnalysisManager &/*MAM*/) {
    bool changed = false;

    // XXX THIS IS HOW YOU QUERY A PLATFORM USING QDMI
    auto supported_gate_set = qdmi_supported_gate_set("Q5");

    // Adding  as requested
    for (auto &function : module){
        auto original_gate = static_cast<std::string>(function.getName());

        bool is_quantum = (original_gate.size() >= QIS_START.size() &&
                           original_gate.substr(0, QIS_START.size()) == QIS_START);

        // We only want to annotate quantum gates
        if (!is_quantum)
            continue;

        // Insert attribute to each unsupported gate
        auto it = std::find(supported_gate_set.begin(), supported_gate_set.end(), original_gate);
        if (it == supported_gate_set.end()) {
            errs() << "              Unsupported gate found: " << original_gate << '\n';
            function.addFnAttr("unsupported");
            changed = true;
        }
    }
    
    if (changed)
        return PreservedAnalyses::none();

    return PreservedAnalyses::all();
}

/**
 * @brief External function for loading the 'QirAnnotateUnsupportedGatesPass' as a 'PassModule'.
 * @return QirAnnotateUnsupportedGatesPass
 */
extern "C" PassModule* loadQirPass() {
    return new QirAnnotateUnsupportedGatesPass();
}
