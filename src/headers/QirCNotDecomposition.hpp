#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirCNotDecompositionPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

