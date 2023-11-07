/**
 * @file PassRunner.cpp
 * @brief TODO
 */

#include "PassRunner.hpp"

/**
 * @brief TODO
 * @param TODO
 */
void invokePasses(std::unique_ptr<Module>  &module,
                  std::vector<std::string>  passes) {
    if (!module) {
        std::cout << "[Pass Runner] Warning: Corrupt QIR module " << std::endl;
        return;
    }

    if (passes.empty()) {
		std::cout << "[Pass Runner] Warning: Not passes found" 
                  << std::endl;
        return;
	}

    // Attach metadata to the IR
    Metadata* metadata = ConstantAsMetadata::get(ConstantInt::get(module->getContext(), 
                                                                  APInt(1, true)));

    module->addModuleFlag(Module::Warning, "lrz_supports_qir", metadata);
    module->setSourceFileName("");

    Metadata* metadataSupport = module->getModuleFlag("lrz_supports_qir");
    if (metadataSupport)
        if (ConstantAsMetadata* boolMetadata = dyn_cast<ConstantAsMetadata>(metadataSupport))
            if (ConstantInt* boolConstant = dyn_cast<ConstantInt>(boolMetadata->getValue()))
                errs() << "[Pass Runner] Flag inserted: \"lrz_supports_qir\" = " 
                       << (boolConstant->isOne() ? "true" : "false") 
                       << '\n';

    // Create an instance of the QirPassRunner and append to it all the received passes
    QirPassRunner &QPR = QirPassRunner::getInstance();
    ModuleAnalysisManager MAM;
    
    for (std::string pass : passes) {
        std::cout << "[Pass Runner] Attempting to invoke pass "
                  << pass
                  << std::endl;

        QPR.append("/usr/local/bin/src/passes/" + pass);
    }

	// Run QIR passes
	QPR.run(*module, MAM);

    // Free memory
    QPR.clearMetadata();

    // Return the adapted QIR's module
    //return module;
}

