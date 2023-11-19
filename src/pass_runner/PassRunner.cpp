/**
 * @file PassRunner.cpp
 * @brief TODO
 */

#include "PassRunner.hpp"

/**
 * @brief TODO
 * @param TODO
 */
void invokePasses(std::unique_ptr<Module> &module,
                  std::vector<std::string> passes) {
  if (!module) {
    std::cout << "[Pass Runner]......Warning: Corrupt QIR module " << std::endl;
    return;
  }

  if (passes.empty()) {
    std::cout << "[Pass Runner]......Warning: Not passes found" << std::endl;
    return;
  }

  // Attach metadata to the IR
  Metadata *metadata = ConstantAsMetadata::get(
      ConstantInt::get(module->getContext(), APInt(1, true)));

  module->addModuleFlag(Module::Warning, "lrz_supports_qir", metadata);
  module->setSourceFileName("");

  Metadata *metadataSupport = module->getModuleFlag("lrz_supports_qir");
  if (metadataSupport)
    if (ConstantAsMetadata *boolMetadata =
            dyn_cast<ConstantAsMetadata>(metadataSupport))
      if (ConstantInt *boolConstant =
              dyn_cast<ConstantInt>(boolMetadata->getValue()))
        errs() << "[Pass Runner]......Flag inserted: \"lrz_supports_qir\" = "
               << (boolConstant->isOne() ? "true" : "false") << '\n';

  // Create an instance of the QirPassRunner and append to it all the received
  // passes
  QirPassRunner &QPR = QirPassRunner::getInstance();
  ModuleAnalysisManager MAM;

  char buffer[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (len == -1) {
    errs() << "[Pass Runner]......Error: Erroneous path to pass\n";
    QPR.clearMetadata();
    return;
  }

  buffer[len] = '\0';

  std::string pathPass = std::string(buffer);
  size_t lastSlash = pathPass.find_last_of("/\\");
  pathPass = pathPass.substr(0, lastSlash) + "/lib/pass_runner/passes/";
  for (std::string pass : passes) {
    std::string libPass = pathPass + pass;
    QPR.append(libPass);
  }

  // Run QIR passes
  QPR.run(*module, MAM);

  // Free memory
  QPR.clearMetadata();

  // Return the adapted QIR's module
  // return module;
}
