#!/bin/bash
set -eu

# Create out directory
mkdir -p bin

# Build and Simulate Adder
iverilog -o bin/adder ../modules/generics.v adder_tb.v
vvp bin/adder

# Build and Simulate Mux2
iverilog -o bin/mux2 ../modules/generics.v mux2_tb.v
vvp bin/mux2

# Visualize
#gtkwave bin/adder_tb.vcd
#gtkwave bin/mux2_tb.vcd