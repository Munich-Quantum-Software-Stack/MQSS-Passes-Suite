/**
 * @file qdmi.hpp
 * @brief Dummy header of the QDMI.
 */

#ifndef QDMI_HPP
#define QDMI_HPP

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../backends/IQMBackend.hpp"

/**
 * @brief Function that returns a vector with the
 * list of availble quantum platforms.
 * @return std::vector<std::string>
 * @todo To be implemented
 */
extern "C" {
std::vector<std::string> qdmi_backend_available_platforms();
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

/**
 * @brief TODO
 * @param TODO
 * @return TODO
 * @todo To be implemented
 */
extern "C" {
std::shared_ptr</*void*/ JobRunner>
qdmi_backend_open(const std::string &target_platform);
}

/**
 * @brief TODO
 * @param TODO
 * @return TODO
 * @todo To be implemented
 */
extern "C" {
void qdmi_launch_qir(std::shared_ptr<JobRunner> handle,
                     const std::string &circuit, int n_shots);
}

/**
 * @brief TODO
 * @param TODO
 * @return TODO
 * @todo To be implemented
 */
extern "C" {
int qdmi_backend_close(std::shared_ptr<JobRunner> handle);
}

#endif // QDMI_HPP
