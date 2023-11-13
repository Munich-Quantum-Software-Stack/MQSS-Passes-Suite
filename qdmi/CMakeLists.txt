cmake_minimum_required(VERSION 3.22.1)
project(qdmi
        VERSION 0.0
        DESCRIPTION "Quantum Device Management Interface")

set(CMAKE_CXX_STANDARD 14)

add_library(qdmi SHARED qdmi.cpp)

target_include_directories(qdmi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

