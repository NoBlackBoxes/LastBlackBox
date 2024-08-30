#!/bin/bash
set -eu

# Create out directory
mkdir -p bin

# Build
iverilog -o bin/adc adc.v modules/rx.v modules/tx.v adc_tb.v

# Simulate
vvp bin/adc

# Visualize
#gtkwave bin/adc_tb.vcd