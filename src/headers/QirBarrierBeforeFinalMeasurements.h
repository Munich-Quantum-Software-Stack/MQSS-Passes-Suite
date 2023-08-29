#pragma once

#include "PassModule.h"

namespace llvm {

class QirBarrierBeforeFinalMeasurementsPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &/*mam*/);
};

}

