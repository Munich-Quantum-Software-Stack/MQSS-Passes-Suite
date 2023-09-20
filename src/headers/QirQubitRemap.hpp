#pragma once

#include "PassModule.hpp"

#include <functional>
#include <unordered_map>
#include <vector>

namespace llvm {

class QirQubitRemapPass : public PassModule {
public:
    enum ResourceType
    {
        None,
        Qubit,
        Result
    };

    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

