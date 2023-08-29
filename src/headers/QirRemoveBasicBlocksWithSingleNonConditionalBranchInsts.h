#pragma once

#include "PassModule.h"
#include <algorithm>

namespace llvm {

class QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &/*mam*/);
};

}

