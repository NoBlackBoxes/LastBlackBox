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

# Build and Simulate Main Decoder
iverilog -o bin/main_decoder ../modules/main_decoder.v main_decoder_tb.v
vvp bin/main_decoder

# Build and Simulate ALU Decoder
iverilog -o bin/alu_decoder ../modules/alu_decoder.v alu_decoder_tb.v
vvp bin/alu_decoder

# Build and Simulate Controller
iverilog -o bin/controller ../modules/controller.v ../modules/main_decoder.v ../modules/alu_decoder.v controller_tb.v
vvp bin/controller

# Build and Simulate ALU
iverilog -o bin/alu ../modules/alu.v alu_tb.v
vvp bin/alu

# Build and Simulate Datapath
iverilog -o bin/datapath ../modules/datapath.v ../modules/flopr.v ../modules/adder.v ../modules/mux2.v ../modules/regfile.v ../modules/extend.v ../modules/alu.v ../modules/mux3.v datapath_tb.v
vvp bin/datapath

# Build and Simulate CPU
iverilog -o bin/cpu ../cpu.v ../modules/datapath.v ../modules/flopr.v ../modules/adder.v ../modules/mux2.v ../modules/regfile.v ../modules/extend.v ../modules/alu.v ../modules/mux3.v ../modules/controller.v ../modules/main_decoder.v ../modules/alu_decoder.v cpu_tb.v
vvp bin/cpu

# Visualize
#gtkwave bin/adder_tb.vcd
#gtkwave bin/extend_tb.vcd
#gtkwave bin/flopr_tb.vcd
#gtkwave bin/flopenr_tb.vcd
#gtkwave bin/mux2_tb.vcd
#gtkwave bin/mux3_tb.vcd
#gtkwave bin/regfile_tb.vcd
#gtkwave bin/main_decoder_tb.vcd
#gtkwave bin/alu_decoder_tb.vcd
#gtkwave bin/controller_tb.vcd
#gtkwave bin/alu_tb.vcd
#gtkwave bin/datapath_tb.vcd
#gtkwave bin/cpu_tb.vcd
