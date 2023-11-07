/**
 * @file QirPlaceIrreversibleGatesInMetadata.hpp
 * @brief Declaration of the 'QirPlaceIrreversibleGatesInMetadataPass' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm {

/**
 * @class QirPlaceIrreversibleGatesInMetadataPass
 * @brief This pass inserts gate reversibility information into metadata.
 */
class QirPlaceIrreversibleGatesInMetadataPass : public PassModule {
public:
    static std::string const QIS_START;

    /**
     * @brief Applies this pass to the QIR's LLVM module.
     *
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @return PreservedAnalyses
     */
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

