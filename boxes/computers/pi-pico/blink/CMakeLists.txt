cmake_minimum_required(VERSION 3.13)

# Initialize the Pico SDK directly
include($ENV{HOME}/NoBlackBoxes/LastBlackBox/_tmp/pico/pico-sdk/pico_sdk_init.cmake)

# Configure CMAKE environment
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Swt Pico board type
set(PICO_BOARD pico_w CACHE STRING "Board type")

project(blink C CXX ASM)

# Initialize the Raspberry Pi Pico SDK
pico_sdk_init()

# Add executable. Default name is the project name, version 0.1
add_executable(blink
    blink.c
)

# pull in common dependencies
target_link_libraries(blink pico_stdlib)

if (PICO_CYW43_SUPPORTED)
    target_link_libraries(blink pico_cyw43_arch_none)
endif()

# create map/bin/hex file etc.
pico_add_extra_outputs(blink)

# add url via pico_set_program_url
