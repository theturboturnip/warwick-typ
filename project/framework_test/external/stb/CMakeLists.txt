cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
cmake_policy(VERSION 3.13)

project(stb LANGUAGES CXX)

add_library(stb STATIC)

target_sources(stb PRIVATE ./src/stb_image_impl.cpp)
target_include_directories(stb PUBLIC include)
