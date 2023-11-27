#include "IQMBackend.hpp"

std::vector<int> Q5Backend::run_job(std::unique_ptr<Module> &module,
                                    int n_shots) {

  std::cout << "   [Backend]...........Running Q5Backend job with " << n_shots
            << " shots." << std::endl;

  return {1, 2, 3, 4, 5};
}

int Q5Backend::close_backend() {
  std::cout << "   [Backend]...........Closing Q5Backend: " << std::endl;

  return 0;
}
