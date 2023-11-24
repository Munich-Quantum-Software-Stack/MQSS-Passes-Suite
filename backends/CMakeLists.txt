cmake_minimum_required(VERSION 3.22.1)
project(
  Backends
  VERSION 0.0
  DESCRIPTION "Available backends")

set(CMAKE_CXX_STANDARD 14)

# Define the library
add_library(Backends SHARED "")

# Define the source files for all backends
target_sources(Backends PRIVATE IQMBackend.cpp Q5Backend.cpp Q20Backend.cpp)

# Specify the include directories for Backends library
target_include_directories(Backends PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link the main executable to the backend library
target_link_libraries(${EXECUTABLE_NAME} PRIVATE Backends)

install(TARGETS Backends
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/lib/backends")
