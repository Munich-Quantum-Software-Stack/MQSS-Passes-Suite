include(FetchContent)

FetchContent_Declare(
    qrm
    GIT_REPOSITORY git@github.com:Munich-Quantum-Software-Stack/QRM.git
    GIT_TAG qmap
)

FetchContent_MakeAvailable(qrm)
