#!/bin/bash
set -eu

# Set Root
LBB_ROOT="/home/kampff/NoBlackBoxes/repos/LastBlackBox"

# Set CPU and Modules folder
CPU=$LBB_ROOT"/boxes/fpgas/hdl/verilog/cpu"
MODULES=$CPU"/modules"

# Create out directory
mkdir -p bin

# Build and Simulate Adder
iverilog -o bin/adder $MODULES/adder.v adder_tb.v
vvp bin/adder

# Build and Simulate Generate Immediate
iverilog -o bin/generate_immediate $MODULES/generate_immediate.v generate_immediate_tb.v
vvp bin/generate_immediate

# Build and Simulate Flopr
iverilog -o bin/flopr $MODULES/flopr.v flopr_tb.v
vvp bin/flopr

# Build and Simulate Flopenr
iverilog -o bin/flopenr $MODULES/flopenr.v flopenr_tb.v
vvp bin/flopenr

# Build and Simulate Mux2
iverilog -o bin/mux2 $MODULES/mux2.v mux2_tb.v
vvp bin/mux2

# Build and Simulate Mux3
iverilog -o bin/mux3 $MODULES/mux3.v mux3_tb.v
vvp bin/mux3

# Build and Simulate Mux4
iverilog -o bin/mux8 $MODULES/mux8.v mux8_tb.v
vvp bin/mux8

# Build and Simulate Regfile
iverilog -o bin/regfile $MODULES/regfile.v regfile_tb.v
vvp bin/regfile

# Build and Simulate Main Decoder
iverilog -o bin/main_decoder $MODULES/main_decoder.v main_decoder_tb.v
vvp bin/main_decoder

# Build and Simulate ALU Decoder
iverilog -o bin/alu_decoder $MODULES/alu_decoder.v alu_decoder_tb.v
vvp bin/alu_decoder

# Build and Simulate Controller
iverilog -o bin/controller $MODULES/controller.v $MODULES/main_decoder.v $MODULES/alu_decoder.v controller_tb.v
vvp bin/controller

# Build and Simulate ALU
iverilog -o bin/alu $MODULES/alu.v alu_tb.v
vvp bin/alu

# Build and Simulate Datapath
iverilog -o bin/datapath $MODULES/datapath.v $MODULES/flopr.v $MODULES/adder.v $MODULES/mux2.v $MODULES/mux3.v $MODULES/regfile.v $MODULES/generate_immediate.v $MODULES/alu.v $MODULES/mux8.v datapath_tb.v
vvp bin/datapath

# Build and Simulate CPU
iverilog -o bin/cpu $CPU/cpu.v $MODULES/datapath.v $MODULES/flopr.v $MODULES/adder.v $MODULES/mux2.v $MODULES/mux3.v $MODULES/regfile.v $MODULES/generate_immediate.v $MODULES/alu.v $MODULES/mux8.v $MODULES/controller.v $MODULES/main_decoder.v $MODULES/alu_decoder.v cpu_tb.v
vvp bin/cpu

# Visualize
#gtkwave bin/adder_tb.vcd
#gtkwave bin/generate_immediate_tb.vcd
#gtkwave bin/flopr_tb.vcd
#gtkwave bin/flopenr_tb.vcd
#gtkwave bin/mux3_tb.vcd
#gtkwave bin/mux8_tb.vcd
#gtkwave bin/regfile_tb.vcd
#gtkwave bin/main_decoder_tb.vcd
#gtkwave bin/alu_decoder_tb.vcd
#gtkwave bin/controller_tb.vcd
#gtkwave bin/alu_tb.vcd
#gtkwave bin/datapath_tb.vcd
#gtkwave bin/cpu_tb.vcd
