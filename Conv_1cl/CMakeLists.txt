# CMakefile
# Alessio Burrello <alessio.burrello@unibo.it>
#
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


cmake_minimum_required(VERSION 3.13)

# Allow benchmark to be built as a standalone library.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
    set(CMAKE_TOOLCHAIN_FILE toolchain-gcc CACHE STRING "Toolchain to use")

    project(Benchmarks LANGUAGES C ASM)
    include(SnitchUtilities)

    add_compile_options(-O3 -g -ffunction-sections)

    # Build the runtime.
    add_subdirectory(../snRuntime snRuntime)
endif()

include_directories(DORY_network/inc)
include_directories(${SNRUNTIME_INCLUDE_DIRS})

add_library(dory DORY_network/src/dory.c)
add_library(layerConv0 DORY_network/src/layerConv0.c)
add_library(layerConv1 DORY_network/src/layerConv1.c)
add_library(layerGemm2_last DORY_network/src/layerGemm2_last.c)
add_library(mem_controller DORY_network/src/mem_controller.c)
add_library(network DORY_network/src/network.c)
add_library(occamy_conv2d_optimized DORY_network/src/occamy_conv2d_optimized.c)
add_library(occamy_conv_dw_naive DORY_network/src/occamy_conv_dw_naive.c)
add_library(occamy_conv_naive DORY_network/src/occamy_conv_naive.c)
add_library(occamy_conv_naive_no_padding DORY_network/src/occamy_conv_naive_no_padding.c)
add_library(occamy_nn_utils DORY_network/src/occamy_nn_utils.c)
add_library(occamy_pool_naive DORY_network/src/occamy_pool_naive.c)


add_snitch_executable(main_network DORY_network/src/test_template.c)
target_link_libraries(main_network dory)
target_link_libraries(main_network layerConv0)
target_link_libraries(main_network layerConv1)
target_link_libraries(main_network layerGemm2_last)
target_link_libraries(main_network mem_controller)
target_link_libraries(main_network network)
target_link_libraries(main_network occamy_conv2d_optimized)
target_link_libraries(main_network occamy_conv_dw_naive)
target_link_libraries(main_network occamy_conv_naive)
target_link_libraries(main_network occamy_conv_naive_no_padding)
target_link_libraries(main_network occamy_nn_utils)
target_link_libraries(main_network occamy_pool_naive)
