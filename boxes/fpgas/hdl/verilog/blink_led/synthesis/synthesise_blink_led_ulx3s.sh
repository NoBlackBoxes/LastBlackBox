#!/bin/bash
set -eu

# Create out directory
mkdir -p bin

# Copy verilog module(s)
cp ../blink_led_ulx3s.v bin/.

# Copy constraints file
cp ../ulx3s_v20.lpf bin/.

# Verify Verilog
apio verify --project-dir=bin --board ulx3s-85f --verbose

# Synthesize
apio build --project-dir=bin --board ulx3s-85f --verbose

# Upload
#apio upload --project-dir=bin --board ulx3s-85f --verbose