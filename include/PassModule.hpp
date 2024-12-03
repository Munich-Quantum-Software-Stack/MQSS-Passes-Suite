/**
 * @file PassModule.hpp
 * @brief Declaration of the 'PassModule' class.
 */

#pragma once

#include "llvm.hpp"
#include <qdmi/client.h>

using namespace llvm;

/**
 * @class PassModule
 * @brief This class derives from 'LLVM::ModulePassManager'. The
 * 'AgnosticPassModule' class is an abstract class. The implementations
 * of the virtual member functions are part of the 'QirPassRunner'
 * derived class.
 */
class AgnosticPassModule
{
  public:
    /**
     * @brief Applies a set of target-agnostic passes to a QIR's LLVM module.
     *
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @return PreservedAnalyses
     */
    virtual PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM) = 0;

    /**
     * @brief Destructor for PassModule.
     *
     * Destroys this abstract class
     */
    virtual ~AgnosticPassModule() {}
};

/**
 * @class SpecificPassModule
 * @brief This class derives from 'LLVM::ModulePassManager'. The
 * 'SpecificPassModule' class is an abstract class. The implementations
 * of the virtual member functions are part of the 'QirPassRunner'
 * derived class.
 */
class SpecificPassModule
{
  public:
    /**
     * @brief Applies a set of target-specific passes to a QIR's LLVM module.
     *
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @param dev The target device.
     * @return PreservedAnalyses
     */
    virtual PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM,
                                  QDMI_Device dev) = 0;

    /**
     * @brief Destructor for PassModule.
     *
     * Destroys this abstract class
     */
    virtual ~SpecificPassModule() {}
};
