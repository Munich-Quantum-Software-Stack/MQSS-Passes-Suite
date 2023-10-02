#pragma once

#include "PassModule.hpp"
#include "../QirPassRunner.hpp"

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm {

class QirAnnotateUnsupportedGatesPass : public PassModule {
public:
    static std::string const QIS_START;

    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

