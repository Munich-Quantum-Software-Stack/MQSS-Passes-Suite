/**
 * @file fomac.cpp
 * @brief Implementation of a dummy FoMaC.
 */

#include "fomac.hpp"

#include <qdmi.hpp>

/**
 * @brief TODO
 * @return TODO
 * @todo To be implemented
 */
extern "C" int fomac_gate_set_size(std::string target_platform) {
  std::vector<std::string> supported_set =
      qdmi_supported_gate_set(target_platform);

  return supported_set.size();
}
