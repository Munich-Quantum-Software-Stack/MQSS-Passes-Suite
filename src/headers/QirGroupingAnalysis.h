#pragma once

#include <llvm/IR/PassManager.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <vector>

#include "PassModule.h"

namespace llvm {

struct GroupAnalysis
{
    std::vector<llvm::BasicBlock*> qc_cc_blocks{};
    std::vector<llvm::BasicBlock*> qc_mc_cc_blocks{};
};

class QirGroupingAnalysisPass : public AnalysisInfoMixin<QirGroupingAnalysisPass> {
public:
    using Result = GroupAnalysis;

    static std::string const QIS_START;
    static std::string const READ_INSTR_START;

    QirGroupingAnalysisPass(QirGroupingAnalysisPass const&) = delete;
    QirGroupingAnalysisPass(QirGroupingAnalysisPass&&) = default;
    ~QirGroupingAnalysisPass() = default;
	Result run(Module &module, ModuleAnalysisManager &mam);
    void runBlockAnalysis(Module &module);

    static bool isRequired();
private:
	std::unordered_set<llvm::BasicBlock*> contains_quantum_circuit_{};
    std::unordered_set<llvm::BasicBlock*> contains_quantum_measurement_{};

    std::unordered_set<llvm::BasicBlock*> pure_quantum_instructions_{};
    std::unordered_set<llvm::BasicBlock*> pure_quantum_measurement_{}; 
	
	static AnalysisKey Key;
    friend struct AnalysisInfoMixin<QirGroupingAnalysisPass>;
};

}
