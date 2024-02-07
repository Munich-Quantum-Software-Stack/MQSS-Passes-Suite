find_package(Doxygen REQUIRED)

set(DOXYGEN_PROJECT_NAME "QIR Passes")
set(DOXYGEN_PROJECT_BRIEF
    "This project consists of QIR Passes."
)

if(BUILD_DOCS_STANDALONE)
  set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR}/../src)
  set(DOXYGEN_OUTPUT_DIR ${CMAKE_SOURCE_DIR})
  set(DOXYGEN_INDEX_FILE ${CMAKE_SOURCE_DIR}/html/index.xml)
else()
  set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR}/src)
  set(DOXYGEN_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/docs)
  set(DOXYGEN_INDEX_FILE ${CMAKE_SOURCE_DIR}/docs/html/index.xml)
endif()
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
add_custom_target(docs_passes ALL DEPENDS ${DOXYGEN_INDEX_FILE})
