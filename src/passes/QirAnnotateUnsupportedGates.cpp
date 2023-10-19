#include "../headers/QirAnnotateUnsupportedGates.hpp"

using namespace llvm;

std::string const QirAnnotateUnsupportedGatesPass::QIS_START = "__quantum"
                                                               "__qis_";

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

extern "C" PassModule* loadQirPass() {
    return new QirAnnotateUnsupportedGatesPass();
}
