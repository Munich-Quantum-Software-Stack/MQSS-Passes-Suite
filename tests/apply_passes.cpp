#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <iostream>
#include <dlfcn.h>

#include "../src/headers/PassModule.h"

using namespace llvm;

int main() {
    PassBuilder             PB;
    LoopAnalysisManager     LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager    CGAM;
    ModuleAnalysisManager   MAM; 

    // Set up the analysis managers
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);

    LLVMContext Context;

    // Step 1: Read the LLVM IR from .ll file
    SMDiagnostic error;
    std::unique_ptr<Module> module = parseIRFile("../benchmarks/bell_state.ll", error, Context);
    if (!module) {
        std::cerr << "Error reading .ll file." << std::endl;
        error.print("apply_pass", errs());
        return 1;
    }

    // Step 2: Load the shared objects and create instances of your passes
    void *soHandle1 = dlopen("./src/passes/libQirRemoveNonEntrypointFunctionsPass.so", RTLD_LAZY);
    if (!soHandle1) {
        std::cerr << "Error loading shared object." << std::endl;
        return 1;
    }

    void *soHandle2 = dlopen("./src/passes/libQirGroupingPass.so", RTLD_LAZY);
    if (!soHandle2) {
        std::cerr << "Error loading shared object." << std::endl;
        return 1;
    }

    void *soHandle3 = dlopen("./src/passes/libQirBarrierBeforeFinalMeasurementsPass.so", RTLD_LAZY);
    if (!soHandle3) {
        std::cerr << "Error loading shared object." << std::endl;
        return 1;
    }
    
    void *soHandle4 = dlopen("./src/passes/libQirCXCancellationPass.so", RTLD_LAZY);
    if (!soHandle4) {
        std::cerr << "Error loading shared object." << std::endl;
        return 1;
    }

    void *soHandle5 = dlopen("./src/passes/libQirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass.so", RTLD_LAZY);
    if (!soHandle5) {
        std::cerr << "Error loading shared object." << std::endl;
        return 1;
    }

    using passCreator = PassModule* (*)();
    
    passCreator createPass1 = reinterpret_cast<passCreator>(dlsym(soHandle1, "createQirRemoveNonEntrypointFunctionsPass"));
    if (!createPass1) {
        std::cerr << "Error getting factory function: " << dlerror() << std::endl;
        dlclose(soHandle1);
        return 1;
    }

    passCreator createPass2 = reinterpret_cast<passCreator>(dlsym(soHandle2, "createQirGroupingPass"));
    if (!createPass2) {
        std::cerr << "Error getting factory function: " << dlerror() << std::endl;
        dlclose(soHandle2);
        return 1;
    }

    passCreator createPass3 = reinterpret_cast<passCreator>(dlsym(soHandle3, "createQirBarrierBeforeFinalMeasurementsPass"));
    if (!createPass3) {
        std::cerr << "Error getting factory function: " << dlerror() << std::endl;
        dlclose(soHandle3);
        return 1;
    }

    passCreator createPass4 = reinterpret_cast<passCreator>(dlsym(soHandle4, "createQirCXCancellationPass"));
    if (!createPass4) {
        std::cerr << "Error getting factory function: " << dlerror() << std::endl;
        dlclose(soHandle4);
        return 1;
    }

    passCreator createPass5 = reinterpret_cast<passCreator>(dlsym(soHandle5, "createQirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass"));
    if (!createPass5) {
        std::cerr << "Error getting factory function: " << dlerror() << std::endl;
        dlclose(soHandle5);
        return 1;
    }

    PassModule *myPass1 = createPass1();
    PassModule *myPass2 = createPass2();
    PassModule *myPass3 = createPass3();
    PassModule *myPass4 = createPass4();
    PassModule *myPass5 = createPass5();

    // Step 3: Apply the pass to the LLVM IR
    myPass1->run(module.get(), MAM);
    myPass2->run(module.get(), MAM);
    myPass3->run(module.get(), MAM);
    myPass4->run(module.get(), MAM);
    myPass5->run(module.get(), MAM);

    // Step 4: Optionally, print the optimized LLVM IR
    module->print(outs(), nullptr);

    // Clean up
    delete myPass1;
    delete myPass2;
    delete myPass3;
    delete myPass4;
    delete myPass5;

    dlclose(soHandle1);
    dlclose(soHandle2);
    dlclose(soHandle3);
    dlclose(soHandle4);
    dlclose(soHandle5);

    return 0;
}

