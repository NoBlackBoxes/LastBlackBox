#!/bin/bash
set -eu

# Create out directory
mkdir -p bin

# Copy verilog module(s)
cp ../blink_led.v bin/.

# Copy constraints file
cp ../upduino.pcf bin/.

# Verify Verilog
apio verify --project-dir=bin --board upduino3 --verbose

# Synthesize
apio build --project-dir=bin --board upduino3 --verbose