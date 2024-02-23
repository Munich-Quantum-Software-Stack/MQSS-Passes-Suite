include(FetchContent)

FetchContent_Declare(
    qrm
    GIT_REPOSITORY git@github.com:Munich-Quantum-Software-Stack/QRM.git
    GIT_TAG wmi-backend
)

FetchContent_MakeAvailable(qrm)
