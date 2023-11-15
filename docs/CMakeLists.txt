find_package(Doxygen REQUIRED)

set(DOXYGEN_PROJECT_NAME "Quantum Resource Manager (QRM) daemon")
set(DOXYGEN_PROJECT_BRIEF "This project consists of the Scheduler Runner, the Selector Runner, and the Pass Runner.")

set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR}/src)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/docs)
set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/html/index.xml)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# Generate DoxyFile from the input file.
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

# Create Output directory.
file(REMOVE_RECURSE ${DOXYGEN_OUTPUT_DIR}/html)
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

# Command for generating doc from Doxygen config file.
add_custom_command(
  OUTPUT ${DOXYGEN_INDEX_FILE}
  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
  MAIN_DEPENDENCY ${DOXYFILE_OUT}
  ${DOXYFILE_IN}
  COMMENT "Generating Doxygen documentation"
  VERBATIM)

# Create CMake Target for generating doc.
add_custom_target(docs ALL DEPENDS ${DOXYGEN_INDEX_FILE})

