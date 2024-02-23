include(FetchContent)

FetchContent_Declare(
    fomac
    GIT_REPOSITORY git@github.com:Munich-Quantum-Software-Stack/FoMaC.git
    GIT_TAG wmi-backend
)

FetchContent_MakeAvailable(fomac)
