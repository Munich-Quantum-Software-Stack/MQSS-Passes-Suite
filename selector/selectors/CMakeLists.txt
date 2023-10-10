# Include all new selectors here
set(SELECTORS_SOURCE_FILES
    selector_all.cpp
    selector_basic.cpp
)

# Compile each selector as a shared object
foreach(src_file ${SELECTORS_SOURCE_FILES})
    get_filename_component(src_name ${src_file} NAME_WE)
    set(target_name "${src_name}")
    add_library(${target_name} MODULE ${src_file})
endforeach()

