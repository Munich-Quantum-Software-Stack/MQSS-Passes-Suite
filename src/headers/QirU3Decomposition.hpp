#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirU3DecompositionPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

