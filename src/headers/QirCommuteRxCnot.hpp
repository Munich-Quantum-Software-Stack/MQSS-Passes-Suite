#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirCommuteRxCnotPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

