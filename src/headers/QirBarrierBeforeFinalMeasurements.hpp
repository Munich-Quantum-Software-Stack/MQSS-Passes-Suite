#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirBarrierBeforeFinalMeasurementsPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

