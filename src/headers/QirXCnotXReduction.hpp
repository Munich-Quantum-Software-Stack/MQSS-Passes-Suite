#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirXCnotXReductionPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

