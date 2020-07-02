cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
cmake_policy(VERSION 3.13)

project(imgui LANGUAGES CXX)

file(GLOB SOURCES src/*.cpp)

find_package(Vulkan REQUIRED)
find_package(SDL2 REQUIRED)

add_library(imgui STATIC)

target_sources(imgui PRIVATE ${SOURCES})
target_include_directories(imgui
        PUBLIC
            include
        PRIVATE
            ${Vulkan_INCLUDE_DIRS}
            ${SDL2_INCLUDE_DIRS})
target_link_libraries(imgui PUBLIC ${Vulkan_LIBRARIES} ${SDL2_LIBRARIES})
