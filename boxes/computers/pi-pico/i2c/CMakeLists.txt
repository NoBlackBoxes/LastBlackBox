cmake_minimum_required(VERSION 3.13)

# Initialize the Pico SDK directly
include($ENV{HOME}/NoBlackBoxes/LastBlackBox/_tmp/pico/pico-sdk/pico_sdk_init.cmake)

# Configure CMAKE environment
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Swt Pico board type
set(PICO_BOARD pico_w CACHE STRING "Board type")

project(i2c C CXX ASM)

# Initialize the Raspberry Pi Pico SDK
pico_sdk_init()

# Add executable. Default name is the project name, version 0.1
add_executable(i2c
    i2c.c
)

# pull in common dependencies
target_link_libraries(i2c pico_stdlib hardware_i2c)

# enable usb output, disable uart output
pico_enable_stdio_usb(i2c 1)
pico_enable_stdio_uart(i2c 0)

if (PICO_CYW43_SUPPORTED)
    target_link_libraries(i2c pico_cyw43_arch_none)
endif()

# create map/bin/hex file etc.
pico_add_extra_outputs(i2c)

# add url via pico_set_program_url
