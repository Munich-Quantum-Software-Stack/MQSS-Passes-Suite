/**
 * @file qdmi.hpp
 * @brief Dummy header of the QDMI.
 */

#ifndef QDMI_HPP
#define QDMI_HPP

#include <string>
#include <vector>

/**
 * @brief Function that returns a vector with the
 * list of availble quantum platforms.
 * @return std::vector<std::string>
 * @todo To be implemented
 */
extern "C" {
std::vector<std::string> qdmi_available_platforms();
}

/**
 * @brief Function that returns a vector with the
 * supported gate set of a given target platform.
 * @param target_platform Target architecture to query.
 * @return std::vector<std::string>
 * @todo To be implemented
 */
extern "C" {
std::vector<std::string> qdmi_supported_gate_set(std::string);
}

#endif // QDMI_HPP
