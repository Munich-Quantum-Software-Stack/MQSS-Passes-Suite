# Include all new selectors here
set(SELECTORS_SOURCE_FILES selector_all.cpp
                           # selector1.cpp ... selectorN.cpp
)

# Set RPATH for the shared libraries in this directory
set(CMAKE_INSTALL_RPATH "${INSTALL_PATH}/bin/lib/selector_runner/selectors")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Compile each selector as a shared object
foreach(src_file ${SELECTORS_SOURCE_FILES})
  get_filename_component(src_name ${src_file} NAME_WE)
  set(target_name "${src_name}")
  add_library(${target_name} MODULE ${src_file})
  install(TARGETS ${target_name}
          DESTINATION "${INSTALL_PATH}/bin/lib/selector_runner/selectors")
endforeach()
