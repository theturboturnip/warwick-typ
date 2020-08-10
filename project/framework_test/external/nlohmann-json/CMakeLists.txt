cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
cmake_policy(VERSION 3.13)

project(nlohmann_json LANGUAGES CXX)

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE include)
target_precompile_headers(nlohmann_json INTERFACE include/nlohmann/json.hpp)