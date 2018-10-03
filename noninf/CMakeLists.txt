cmake_minimum_required(VERSION 2.8.12)
project(SVI)

include_directories(include/)
include_directories(/usr/lib64/boost)

set(CMAKE_CXX_FLAGS "-O3")


add_executable(noninf src/noninf.cpp src/genotype.cpp src/storage.cpp)
