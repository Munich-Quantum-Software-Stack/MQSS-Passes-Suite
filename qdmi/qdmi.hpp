#ifndef QDMI_HPP
#define QDMI_HPP

#include <vector>
#include <string>

std::vector<std::string> qdmi_available_platforms();
std::vector<std::string> qdmi_supported_gate_set(std::string);

#endif // QDMI_HPP

