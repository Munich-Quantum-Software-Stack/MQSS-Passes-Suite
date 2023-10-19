/**
 * The 'PassModule' class is an abstract class. The implementations
 * of the virtual member functions are part of the 'QirPassRunner' 
 * derived class.
 */

#pragma once

#include "llvm.hpp"
#include <qdmi.hpp>
#include "../QirPassRunner.hpp"

using namespace llvm;

class PassModule {
public:
    /**
     * Applies a set of passes to a QIR's LLVM module
     */
    virtual PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM) = 0;

    /**
     * Destroys this abstract class
     */
    virtual ~PassModule() {}
};

