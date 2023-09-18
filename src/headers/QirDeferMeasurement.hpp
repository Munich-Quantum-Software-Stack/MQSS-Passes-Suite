#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirDeferMeasurementPass : public PassModule {
public:
    static std::string const RECORD_INSTR_END;
    QirDeferMeasurementPass();
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
private:
    std::unordered_set<std::string> readout_names_{};
};

}

