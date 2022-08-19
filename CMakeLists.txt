cmake_minimum_required(VERSION 3.21)
project(spectral_clustering C)

set(CMAKE_C_STANDARD 90)

add_executable(spectral_clustering spkmeans.c tests.c)
