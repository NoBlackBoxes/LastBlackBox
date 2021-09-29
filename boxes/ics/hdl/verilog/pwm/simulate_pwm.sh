#!/bin/bash
set -eu

# Create out directory
mkdir -p bin

# Build
iverilog -o bin/pwm pwm.v pwm_tb.v

# Simulate
vvp bin/pwm

# Visualize
gtkwave bin/pwm_tb.vcd