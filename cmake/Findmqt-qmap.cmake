include(FetchContent)

FetchContent_Declare(
    mqt-qmap
    #GIT_REPOSITORY git@github.com:cda-tum/mqt-qmap.git
    #GIT_TAG aae9e37f551a830a99b8c1d36d05208c674dd2db
    SOURCE_DIR /home/ubuntu/mqt-qmap
)

FetchContent_MakeAvailable(mqt-qmap)

FetchContent_GetProperties(mqt-qmap)

set(QMAP_INCLUDE_DIRS "${mqt-qmap_SOURCE_DIR}/include")
