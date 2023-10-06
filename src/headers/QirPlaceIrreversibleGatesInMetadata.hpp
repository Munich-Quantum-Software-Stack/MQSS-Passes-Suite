#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirPlaceIrreversibleGatesInMetadataPass : public PassModule {
public:
    static std::string const QIS_START;

    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

