#!/bin/bash
set -eu

# Create out directory
mkdir -p bin

# Build
iverilog -o bin/sr_latch sr_latch.v sr_latch_tb.v

# Simulate
vvp bin/sr_latch

# Visualize
gtkwave bin/sr_latch_tb.vcd