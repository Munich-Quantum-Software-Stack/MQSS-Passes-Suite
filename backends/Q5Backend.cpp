#include "IQMBackend.hpp"
#include <iostream>

std::vector<int> Q5Backend::run_job(std::unique_ptr<Module> &module,
                                    int n_shots) {
  std::cout << "Running Q5Backend job with " << n_shots << " shots."
            << std::endl;

  return {1, 2, 3, 4, 5};
}

int Q5Backend::close_backend() {
  std::cout << "Closing Q5Backend: " << std::endl;
  return 0;
}
