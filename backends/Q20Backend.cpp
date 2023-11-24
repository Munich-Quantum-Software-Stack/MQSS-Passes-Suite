#include "IQMBackend.hpp"
#include <iostream>

void Q20Backend::run_job(const std::string &circuit, int n_shots) {
  std::cout << "Running Q20Backend job: " << circuit << " with " << n_shots
            << " shots." << std::endl;
}

int Q20Backend::close_backend() {
  std::cout << "Closing Q20Backend: " << std::endl;
  return 0;
}
