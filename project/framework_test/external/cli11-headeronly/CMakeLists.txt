cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
cmake_policy(VERSION 3.13)

project(CLI11 LANGUAGES CXX)

add_library(CLI11 INTERFACE)
target_include_directories(CLI11 INTERFACE include)
