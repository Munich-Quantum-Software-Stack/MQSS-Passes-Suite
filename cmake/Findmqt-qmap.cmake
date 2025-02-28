include(FetchContent)

FetchContent_Declare(
     mqt-qmap
     GIT_REPOSITORY git@github.com:cda-tum/mqt-qmap.git
     GIT_TAG fa598ff831d5e598e2ff9748137b86cbddad6f28
)
 
FetchContent_MakeAvailable(mqt-qmap)
 
FetchContent_GetProperties(mqt-qmap)
 
set(QMAP_INCLUDE_DIRS "${mqt-qmap_SOURCE_DIR}/include")
