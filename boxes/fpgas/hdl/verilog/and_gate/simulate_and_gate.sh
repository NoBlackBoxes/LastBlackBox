#!/bin/bash
set -eu

# Create output directory
mkdir -p bin

# Build
iverilog -o bin/and_gate and_gate.v and_gate_tb.v

# Simulate
vvp bin/and_gate

# Visualize
gtkwave bin/and_gate_tb.vcd