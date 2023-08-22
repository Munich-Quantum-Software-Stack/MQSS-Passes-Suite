// QIR Pass Manager
#include "../src/QirPassManager.h"

// Passes
#include "../src/headers/QirGrouping.h"
#include "../src/headers/QirBarrierBeforeFinalMeasurements.h"
#include "../src/headers/QirRemoveBasicBlocksWithSingleNonConditionalBranchInsts.h"
#include "../src/headers/QirCXCancellation.h"
#include "../src/headers/QirRemoveNonEntrypointFunctions.h"

#include <iostream>

using namespace llvm;

int main() {
    ModuleAnalysisManager MAM;
    QirPassManager        QPM;
    LLVMContext           Context;
    SMDiagnostic          error;
    
    // Read QIR
    std::unique_ptr<Module> module = parseIRFile("../benchmarks/bell_state.ll", error, Context);
    if(!module) {
        std::cerr << "Error reading .ll file." << std::endl;
        return 1;
    }

    // Append passes
    QPM.append(new QirRemoveNonEntrypointFunctionsPass());
    QPM.append(new QirGroupingPass());
    QPM.append(new QirBarrierBeforeFinalMeasurementsPass());
    QPM.append(new QirCXCancellationPass());
    QPM.append(new QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass());

    // Run the passes
    QPM.run(module.get(), MAM);

    // Print the result
    module->print(outs(), nullptr);

    // Free memory
    for(PassModule* pass : QPM.getPasses())
        delete pass;

    return 0;
}

