cmake_minimum_required(VERSION 4.0)
project(gig_smoothing_by_diffusion)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

FetchContent_Declare(
        OpenMesh
        GIT_REPOSITORY https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh.git
        GIT_TAG OpenMesh-11.0
)

# Make OpenMesh available
FetchContent_MakeAvailable(OpenMesh)

FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
)

FetchContent_MakeAvailable(Eigen)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
        OpenMeshCore
        OpenMeshTools

        Eigen3::Eigen
)
