cmake_minimum_required(VERSION 3.22.1)
project(QDMI
        VERSION 0.0
        DESCRIPTION "Quantum Device Management Interface")

set(CMAKE_CXX_STANDARD 14)

add_library(QDMI SHARED qdmi.cpp)

target_include_directories(QDMI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

