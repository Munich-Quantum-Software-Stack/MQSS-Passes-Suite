#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirRzDecompositionPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

