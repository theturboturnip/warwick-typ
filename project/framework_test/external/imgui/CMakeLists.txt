cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
cmake_policy(VERSION 3.13)

project(imgui LANGUAGES CXX)

file(GLOB SOURCES src/*.cpp)

# Create imgui as an INTERFACE library, which means the sources are compiled with each project that includes them.
# This is simpler than a STATIC library, which would need to be linked to Vulkan and SDL2 at this point.
add_library(imgui INTERFACE)

target_sources(imgui INTERFACE ${SOURCES})
target_include_directories(imgui INTERFACE include)
