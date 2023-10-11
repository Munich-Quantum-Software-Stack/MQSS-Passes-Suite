#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirCommuteCnotRxPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

