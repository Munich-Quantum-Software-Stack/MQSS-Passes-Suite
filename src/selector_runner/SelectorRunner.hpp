/**
 * @file SelectorRunner.hpp
 * @brief TODO
 */

#ifndef SELECTORRUNNER_HPP
#define SELECTORRUNNER_HPP

#include <iostream>
#include <cstring>
#include <csignal>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <unistd.h>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>
#include <dlfcn.h>

std::vector<std::string> invokeSelector(const char *pathSelector);

#endif // SELECTORRUNNER_HPP
