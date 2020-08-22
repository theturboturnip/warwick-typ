cmake_minimum_required(VERSION 3.16)
project(framework_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_PROPAGATE_HOST_FLAGS FALSE)
else()
    message(WARNING "CUDA not found, disabling...")
endif()

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR "This project must be compiled with GCC for correct results")
endif()

# Set up resource copying
add_custom_target(copy_assets ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/assets
        ${CMAKE_BINARY_DIR}
        )

string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)
if (BUILD_TYPE_LOWER MATCHES ".*debug.*")
    set(IS_DEBUG TRUE)
else()
    set(IS_DEBUG FALSE)
endif()

# Set up CXX flags
# TODO: -O2 on debug?
# NOTE: Right now the simple sim is compiled with vectorization, so auto-vectorization will take place
# These work on GCC-8 with the optimized version
set(CXX_SIMD_OPT -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -ffp-contract=off)
set(CXX_WARNING_OPT -Wall -Werror -Wextra -Wno-error=extra -Wno-error=unused-parameter -Wno-error=unused-result -Wno-error=unused-but-set-variable -Wno-error=unused-variable -Wno-error=comment)
if (IS_DEBUG)
    message("DEBUG MODE")
    set(CXX_GENERAL_OPT -g3 -Og) # -fsanitize=address)
    set(CXX_DEBUG_LIBS)# -lasan)
    set(CUDA_DEBUG_OPT -G)
else()
    set(CXX_GENERAL_OPT -flto -O2)
    set(CXX_DEBUG_LIBS)
    set(CUDA_DEBUG_OPT)
endif()
set(CXX_OVERALL_OPT ${CXX_SIMD_OPT} ${CXX_WARNING_OPT} ${CXX_GENERAL_OPT})
set(CUDA_OVERALL_OPT --extended-lambda ${CUDA_DEBUG_OPT})# -G -arch compute_70)

# Set up external dependencies
add_subdirectory(external)
find_package(Vulkan REQUIRED)
find_package(SDL2 REQUIRED)
find_package(OpenMP REQUIRED)

# Create a list of libraries used in both CUDA and non-CUDA
set(GLOBAL_LINKED_LIBS
        ${Vulkan_LIBRARIES} ${SDL2_LIBRARIES}
        OpenMP::OpenMP_CXX glm imgui GSL CLI11 nlohmann_json stb
        ${CXX_DEBUG_LIBS}
        )
set(GLOBAL_INCLUDE_DIRS ./src/)
set(GLOBAL_SYSTEM_INCLUDE_DIRS ${Vulkan_INCLUDE_DIRS} ${SDL2_INCLUDE_DIRS})

# Setup the executables
add_subdirectory(src)

add_custom_target(all_sims)

add_executable(sim_nocuda)
target_link_libraries(sim_nocuda PRIVATE generic_sources)
target_link_libraries(sim_nocuda PRIVATE ${GLOBAL_LINKED_LIBS})
target_include_directories(sim_nocuda SYSTEM PRIVATE ${GLOBAL_SYSTEM_INCLUDE_DIRS})
target_include_directories(sim_nocuda PRIVATE ${GLOBAL_INCLUDE_DIRS})
target_compile_options(sim_nocuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_OVERALL_OPT}> )
add_dependencies(sim_nocuda copy_assets)
add_dependencies(all_sims sim_nocuda)

if (CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA found, will generate sim_cuda binary")
    add_executable(sim_cuda)
    target_link_libraries(sim_cuda PRIVATE generic_sources cuda_sources)
    target_link_libraries(sim_cuda PRIVATE ${GLOBAL_LINKED_LIBS})
    target_include_directories(sim_cuda SYSTEM PRIVATE ${GLOBAL_SYSTEM_INCLUDE_DIRS})
    target_include_directories(sim_cuda PRIVATE ${GLOBAL_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_options(sim_cuda PRIVATE -DCUDA_ENABLED)
    target_compile_options(sim_cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_OVERALL_OPT}> $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OVERALL_OPT}>)
    add_dependencies(sim_cuda copy_assets)
    add_dependencies(all_sims sim_cuda)
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
