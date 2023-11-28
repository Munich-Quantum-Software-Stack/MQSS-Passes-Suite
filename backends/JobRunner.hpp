#ifndef JOBRUNNER_HPP
#define JOBRUNNER_HPP

#include <chrono>
#include <iostream>
#include <llvm/IR/Module.h>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

using namespace llvm;

class JobRunner
{
  public:
    virtual int close_backend() { return 0; };
    virtual std::unordered_map<std::string, int>
    run_job(std::unique_ptr<Module> &module, int n_shots) = 0;
    virtual ~JobRunner() {}
};

#endif /* JOBRUNNER_HPP */
