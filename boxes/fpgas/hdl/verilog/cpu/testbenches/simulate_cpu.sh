#!/bin/bash
set -eu

# Create out directory
mkdir -p bin

# Build and Simulate Adder
iverilog -o bin/adder ../modules/adder.v adder_tb.v
vvp bin/adder

# Build and Simulate Extend
iverilog -o bin/extend ../modules/extend.v extend_tb.v
vvp bin/extend

# Build and Simulate Flopr
iverilog -o bin/flopr ../modules/flopr.v flopr_tb.v
vvp bin/flopr

# Build and Simulate Flopenr
iverilog -o bin/flopenr ../modules/flopenr.v flopenr_tb.v
vvp bin/flopenr

# Build and Simulate Mux2
iverilog -o bin/mux2 ../modules/mux2.v mux2_tb.v
vvp bin/mux2

# Build and Simulate Mux3
iverilog -o bin/mux3 ../modules/mux3.v mux3_tb.v
vvp bin/mux3

# Build and Simulate Regfile
iverilog -o bin/regfile ../modules/regfile.v regfile_tb.v
vvp bin/regfile

# Visualize
#gtkwave bin/adder_tb.vcd
#gtkwave bin/extend_tb.vcd
#gtkwave bin/flopr_tb.vcd
#gtkwave bin/flopenr_tb.vcd
#gtkwave bin/mux2_tb.vcd
#gtkwave bin/mux3_tb.vcd
gtkwave bin/regfile_tb.vcd
