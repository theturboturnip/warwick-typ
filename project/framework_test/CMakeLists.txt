cmake_minimum_required(VERSION 3.16)
project(framework_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_PROPAGATE_HOST_FLAGS FALSE)
else()
    message(WARNING "CMake not found, disabling...")
endif()

add_subdirectory(external)

find_package(Vulkan REQUIRED)
find_package(SDL2 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(framework_test)

file(GLOB_RECURSE CXX_SOURCES ./src/*.cpp ./src/*.h ./src/*.inl)
target_sources(framework_test PRIVATE ${CXX_SOURCES})
target_include_directories(framework_test PRIVATE ./src/)
if (CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE CUDA_SOURCES ./src/*.cu)
    target_sources(framework_test PRIVATE ${CUDA_SOURCES})
endif()

#if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using Clang
    # set(CXX_WARNING_OPT -Wall -Werror -Wno-error-unused)
#elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # using GCC
#elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # using Intel C++
#elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
#end()

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR "This project must be compiled with GCC for correct results")
endif()

# TODO: -O2 on debug
# NOTE: Right now the simple sim is compiled with vectorization, so auto-vectorization will take place
# These work on GCC-8 with the optimized version
set(CXX_SIMD_OPT -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -ffp-contract=off -fno-fast-math)
set(CXX_WARNING_OPT -Wall -Werror -Wno-error=unused -Wno-error=unused-result -Wno-error=unused-but-set-variable -Wno-error=unused-variable)

target_compile_options(framework_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++14 -flto ${CXX_WARNING_OPT} ${CXX_SIMD_OPT}>)
if (CMAKE_CUDA_COMPILER)
    target_compile_options(framework_test PRIVATE -DCUDA_ENABLED)
endif()

target_include_directories(framework_test SYSTEM PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(framework_test PRIVATE ${Vulkan_LIBRARIES})

target_include_directories(framework_test SYSTEM PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(framework_test PRIVATE ${SDL2_LIBRARIES})

target_link_libraries(framework_test PUBLIC OpenMP::OpenMP_CXX)

# This project uses a local installation of GLM so find_package isn't necessary
target_link_libraries(framework_test PRIVATE glm)

target_link_libraries(framework_test PRIVATE imgui)

target_link_libraries(framework_test PRIVATE GSL)

# Set up resource copying
add_custom_target(copy_assets ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/assets
        ${CMAKE_BINARY_DIR}
        )
add_dependencies(framework_test copy_assets)
