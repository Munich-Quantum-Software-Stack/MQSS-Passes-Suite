#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirNormalizeArgAnglePass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

