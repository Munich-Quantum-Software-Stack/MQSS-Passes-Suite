/**
 * @file QirPlaceIrreversibleGatesInMetadata.cpp
 * @brief Implementation of the 'QirPlaceIrreversibleGatesInMetadataPass' class.
 * <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirPlaceIrreversibleGatesInMetadata.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from:
 */

#include "../headers/QirPlaceIrreversibleGatesInMetadata.hpp"

using namespace llvm;

/**
 * @var QirPlaceIrreversibleGatesInMetadataPass::QIS_START
 * @brief Used within the 'QirPlaceIrreversibleGatesInMetadataPass'
 * to define the quantum prefix.
 */
std::string const QirPlaceIrreversibleGatesInMetadataPass::QIS_START =
    "__quantum"
    "__qis_";

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses
QirPlaceIrreversibleGatesInMetadataPass::run(Module &module,
                                             ModuleAnalysisManager &MAM) {
  QirPassRunner &QPR = QirPassRunner::getInstance();
  QirMetadata &qirMetadata = QPR.getMetadata();

  for (auto &function : module) {
    auto name = static_cast<std::string>(function.getName());
    bool is_quantum = (name.size() >= QIS_START.size() &&
                       name.substr(0, QIS_START.size()) == QIS_START);

    if (is_quantum) {
      auto name = static_cast<std::string>(function.getName());
      if (!function.hasFnAttribute("irreversible")) {
        qirMetadata.append(REVERSIBLE_GATE, name);
        errs() << "   [Pass]..............Reversible gate found: " << name
               << '\n';
      } else {
        qirMetadata.append(IRREVERSIBLE_GATE, name);
        errs() << "   [Pass]..............Irreversible gate found: " << name
               << '\n';
      }
    }
  }

  QPR.setMetadata(qirMetadata);

  return PreservedAnalyses::all();
}

/**
 * @brief External function for loading the
 * 'QirPlaceIrreversibleGatesInMetadataPass' as a 'PassModule'.
 * @return QirPlaceIrreversibleGatesInMetadataPass
 */
extern "C" PassModule *loadQirPass() {
  return new QirPlaceIrreversibleGatesInMetadataPass();
}
