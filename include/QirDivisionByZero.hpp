/**
 * @file QirDivisionByZero.hpp
 * @brief Declaration of the 'QirDivisionByZeroPass' class.
 */

#pragma once

#include <PassModule.hpp>

#include <functional>
#include <unordered_map>
#include <vector>

namespace llvm
{

/**
 * @class QirDivisionByZeroPass
 * @brief TODO
 */
class QirDivisionByZeroPass : public AgnosticPassModule
{
  public:
    static const char *const EC_VARIABLE_NAME;    /**< TODO */
    static const char *const EC_REPORT_FUNCTION;  /**< TODO */
    static int64_t const EC_QIR_DIVISION_BY_ZERO; /**< TODO */

    /**
     * @brief TODO
     */
    QirDivisionByZeroPass() = default;

    /**
     * @brief TODO
     */
    QirDivisionByZeroPass(QirDivisionByZeroPass const &) = delete;

    /**
     * @brief TODO
     */
    QirDivisionByZeroPass(QirDivisionByZeroPass &&) = default;

    /**
     * @brief TODO
     */
    ~QirDivisionByZeroPass() = default;

    /**
     * @brief Applies this pass to the QIR's LLVM module.
     *
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @return PreservedAnalyses
     */
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);

    /**
     * @brief TODO
     * @param error_code TODO
     * @param module TODO
     * @param instr TODO
     */
    void raiseError(int64_t error_code, Module &module, Instruction *instr);

  private:
    GlobalVariable *error_variable_{nullptr};
};

} // namespace llvm
