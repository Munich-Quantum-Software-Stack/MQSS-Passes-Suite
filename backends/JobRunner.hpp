#ifndef JOBRUNNER_HPP
#define JOBRUNNER_HPP

#include <llvm/IR/Module.h>
#include <memory>
#include <string>

using namespace llvm;

class JobRunner {
public:
  virtual int close_backend() { return 0; };
  virtual std::vector<int> run_job(std::unique_ptr<Module> &module,
                                   int n_shots) = 0;
  virtual ~JobRunner() {}
};

#endif /* JOBRUNNER_HPP */
