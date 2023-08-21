cmake_minimum_required(VERSION 3.20)
project(QirPassManager
  VERSION 1.0
  DESCRIPTION "QIR Pass Manager")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Set the paths to LLVM libraries and headers
# You can use llvm-config to obtain the necessary flags
find_package(LLVM REQUIRED CONFIG)

# Add the LLVM include directories
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Build your executable
#add_executable(main 
#    tests/main.cpp
#    src/QirPassManager.cpp
#    src/passes/QirBarrierBeforeFinalMeasurements.cpp
#    src/passes/QirCXCancellation.cpp
#    src/passes/QirGrouping.cpp
#    src/passes/QirRemoveBasicBlocksWithSingleNonConditionalBranchInsts.cpp
#    src/passes/QirRemoveNonEntrypointFunctions.cpp
#)

# Add the source files for compilation
file(GLOB SOURCES
    src/QirPassManager.cpp
    src/passes/*.cpp
    tests/main.cpp
)

# Add the sources to the executable
add_executable(main ${SOURCES})

# Link against LLVM libraries
target_link_libraries(main
    PRIVATE
    LLVM
    #LLVMCore
    #LLVMSupport
)

# Print the include directories and libraries found
message("LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message("LLVM_DEFINITIONS: ${LLVM_DEFINITIONS}")

# Print the LLVM libraries being linked
get_target_property(LLVM_LIBS main LINK_LIBRARIES)
message("LLVM_LIBS: ${LLVM_LIBS}")
