#ifndef IQMBACKEND_HPP
#define IQMBACKEND_HPP

#include "JobRunner.hpp"

class IQMBackend : public JobRunner
{
  public:
    virtual int close_backend() override;
    virtual std::unordered_map<std::string, int>
    run_job(std::unique_ptr<Module> &module, int n_shots) override;

    static std::shared_ptr<IQMBackend> create_instance(const std::string &url);
};

// Declarations for Q5Backend
class Q5Backend : public IQMBackend
{
  public:
    std::unordered_map<std::string, int>
    run_job(std::unique_ptr<Module> &module, int n_shots) override;
    int close_backend() override;
};

// Declarations for Q20Backend
class Q20Backend : public IQMBackend
{
  public:
    std::unordered_map<std::string, int>
    run_job(std::unique_ptr<Module> &module, int n_shots) override;
    int close_backend() override;
};

#endif /* IQMBACKEND_HPP */
