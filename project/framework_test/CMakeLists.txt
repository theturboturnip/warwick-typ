cmake_minimum_required(VERSION 3.16)
project(framework_test LANGUAGES CXX CUDA)

#include(CheckLanguage)
#check_language(CUDA)
#if(CMAKE_CUDA_HOST_COMPILER)
#    enable_language(CUDA)
#    message(STATUS "CUDA Enabled")
#endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(SDL2 REQUIRED)
add_subdirectory(external)


add_executable(framework_test)

target_sources(framework_test PRIVATE main.cpp)

target_compile_options(framework_test PRIVATE -Wall -Werror -Wno-deprecated-declarations)

target_include_directories(framework_test PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(framework_test PRIVATE ${Vulkan_LIBRARIES})

target_include_directories(framework_test PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(framework_test PRIVATE ${SDL2_LIBRARIES})

# This project uses a local installation of GLM so find_package isn't necessary
target_link_libraries(framework_test PRIVATE glm)

target_link_libraries(framework_test PRIVATE imgui)
