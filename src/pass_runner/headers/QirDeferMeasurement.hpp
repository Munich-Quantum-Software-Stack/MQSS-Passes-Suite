/**
 * @file QirDeferMeasurement.hpp
 * @brief Declaration of the 'QirDeferMeasurementPass' class.
 */
#pragma once

#include "PassModule.hpp"

namespace llvm {

/**
 * @class QirDeferMeasurementPass
 * @brief This pass moves all measurements to the end of the circuit.
 */
class QirDeferMeasurementPass : public PassModule {
public:
  static std::string const RECORD_INSTR_END;

  /**
   * @brief Constructor for QirDeferMeasurementPass.
   *
   * This constructor initializes the QirDeferMeasurementPass object.
   */
  QirDeferMeasurementPass();

  /**
   * @brief Applies this pass to the QIR's LLVM module.
   *
   * @param module The module of the submitted QIR.
   * @param MAM The module analysis manager.
   * @return PreservedAnalyses
   */
  PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);

private:
  std::unordered_set<std::string> readout_names_{};
};

} // namespace llvm
