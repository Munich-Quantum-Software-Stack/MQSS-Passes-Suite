#ifndef JOBRUNNER_HPP
#define JOBRUNNER_HPP

#include <memory>
#include <string>

class JobRunner {
public:
  virtual int close_backend() { return 0; };
  virtual void run_job(const std::string &circuit, int n_shots) = 0;
  virtual ~JobRunner() {}
};

#endif /* JOBRUNNER_HPP */
