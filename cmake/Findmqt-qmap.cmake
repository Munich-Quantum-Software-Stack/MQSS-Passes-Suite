include(FetchContent)

FetchContent_Declare(
    mqt-qmap
    GIT_REPOSITORY git@github.com:cda-tum/mqt-qmap.git
    GIT_TAG main
)

FetchContent_MakeAvailable(mqt-qmap)

FetchContent_GetProperties(mqt-qmap)

set(QMAP_INCLUDE_DIRS "${mqt-qmap_SOURCE_DIR}/include")
