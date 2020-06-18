cmake_minimum_required(VERSION 3.16)
project(framework_test LANGUAGES CXX CUDA)

set(CUDA_PROPAGATE_HOST_FLAGS FALSE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external)


find_package(Vulkan REQUIRED)
find_package(SDL2 REQUIRED)

add_executable(framework_test)

file(GLOB_RECURSE SOURCES ./src/*.cpp ./src/*.h ./src/*.cu)
target_sources(framework_test PRIVATE ${SOURCES})
target_include_directories(framework_test PRIVATE ./src/)

target_compile_options(framework_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Werror -Wno-deprecated-declarations>)

target_include_directories(framework_test PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(framework_test PRIVATE ${Vulkan_LIBRARIES})

target_include_directories(framework_test PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(framework_test PRIVATE ${SDL2_LIBRARIES})

# This project uses a local installation of GLM so find_package isn't necessary
target_link_libraries(framework_test PRIVATE glm)

target_link_libraries(framework_test PRIVATE imgui)
