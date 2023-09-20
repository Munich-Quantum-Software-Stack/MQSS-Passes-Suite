#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirResourceAnnotationPass : public PassModule {
public:
    enum ResourceType {
        None,
        Qubit,
        Result
    };

    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

