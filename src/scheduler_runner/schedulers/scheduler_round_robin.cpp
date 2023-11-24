/**
 * @file scheduler_round_robin.cpp
 * @brief Implementation of a dummy scheduler.
 */

#include <iostream>
#include <string>
#include <vector>

#include "PassModule.hpp"

#include <fomac.hpp>
#include <qdmi.hpp>

/**
 * @brief The main entry point of the program.
 *
 * The Scheduler.
 *
 * @return const char *
 */
extern "C" void scheduler(void) {
  // Query the available platforms
  std::vector<std::string> platforms = qdmi_backend_available_platforms();

  std::cout << "[Scheduler]........Writing target architecture in the "
               "metadata"
            << std::endl;

  // Choose the target architecture
  QirPassRunner &QPR = QirPassRunner::getInstance();
  QirMetadata &qirMetadata = QPR.getMetadata();
  qirMetadata.setTargetPlatform(platforms.back());
  QPR.setMetadata(qirMetadata);
}
