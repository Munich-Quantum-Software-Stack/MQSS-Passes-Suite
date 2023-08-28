#pragma once
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include <llvm/Support/raw_ostream.h>

#include "PassModule.h"

#include <functional>
#include <unordered_map>
#include <vector>

namespace llvm {

class QirDivisionByZeroPass :  public PassModule {  

    public:
     static const char* const EC_VARIABLE_NAME;
     static const char* const EC_REPORT_FUNCTION;
     static int64_t const     EC_QIR_DIVISION_BY_ZERO;

    QirDivisionByZeroPass() = default;

   // Copy construction is banned.
    QirDivisionByZeroPass(QirDivisionByZeroPass const&) = delete;
   
    // We allow move semantics.
    QirDivisionByZeroPass(QirDivisionByZeroPass&&) = default;
   
    // Default destruction.
   ~QirDivisionByZeroPass() = default;
				     //
   PreservedAnalyses run(Module &module, ModuleAnalysisManager& /*mam*/);
    

    void raiseError(int64_t error_code, Module &module, Instruction* instr);
  
   private:

      GlobalVariable* error_variable_{nullptr};

   };
				     
 } // llvm

