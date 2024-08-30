#!/bin/bash
set -eu

# Create out directory
mkdir -p bin

# Build
iverilog -o bin/counter counter.v counter_tb.v

# Simulate
vvp bin/counter

# Visualize
gtkwave bin/counter_tb.vcd