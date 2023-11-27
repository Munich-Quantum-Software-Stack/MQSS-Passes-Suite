/**
 * @file PassModule.hpp
 * @brief Declaration of the 'PassModule' class.
 */

#pragma once

#include "llvm.hpp"
#include <qdmi.hpp>

#include "../QirPassRunner.hpp"

using namespace llvm;

/**
 * @class PassModule
 * @brief This class derives from 'LLVM::ModulePassManager'. The
 * 'PassModule' class is an abstract class. The implementations
 * of the virtual member functions are part of the 'QirPassRunner'
 * derived class.
 */
class PassModule
{
  public:
    /**
     * @brief Applies a set of passes to a QIR's LLVM module.
     *
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @return PreservedAnalyses
     */
    virtual PreservedAnalyses run(Module &module,
                                  ModuleAnalysisManager &MAM) = 0;

    /**
     * @brief Destructor for PassModule.
     *
     * Destroys this abstract class
     */
    virtual ~PassModule() {}
};
