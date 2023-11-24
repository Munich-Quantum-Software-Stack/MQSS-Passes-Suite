#include "IQMBackend.hpp"
#include <iostream>

void Q5Backend::run_job(const std::string &circuit, int n_shots) {
  std::cout << "Running Q5Backend job: " << circuit << " with " << n_shots
            << " shots." << std::endl;
}

int Q5Backend::close_backend() {
  std::cout << "Closing Q5Backend: " << std::endl;
  return 0;
}
