#!/bin/bash
set -eu

# Create out directory
mkdir -p bin

# Build
iverilog -o bin/full_adder full_adder.v full_adder_tb.v

# Simulate
vvp bin/full_adder

# Visualize
gtkwave bin/full_adder_tb.vcd