#pragma once

#include "PassModule.hpp"

#include <algorithm>

namespace llvm {

class QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

