#include "IQMBackend.hpp"
#include <iostream>

std::vector<int> Q20Backend::run_job(std::unique_ptr<Module> &module,
                                     int n_shots) {
  std::cout << "Running Q20Backend job with " << n_shots << " shots."
            << std::endl;

  return {1, 2, 3, 4, 5};
}

int Q20Backend::close_backend() {
  std::cout << "Closing Q20Backend: " << std::endl;
  return 0;
}
