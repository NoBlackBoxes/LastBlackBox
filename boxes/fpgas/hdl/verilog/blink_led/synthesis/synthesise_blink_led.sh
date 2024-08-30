#!/bin/bash
set -eu

# Set Roots
NBB_ROOT="/home/${USER}/NoBlackBoxes"
LBB_ROOT=$NBB_ROOT"/LastBlackBox"

# Create out directory
mkdir -p bin

# Copy verilog module(s)
cp ../blink_led.v bin/.

# Copy constraints file
cp NB3_hindbrain.pcf bin/.

# Verify Verilog
apio verify --project-dir=bin --board NB3_hindbrain --verbose

# Synthesize
apio build --project-dir=bin --board NB3_hindbrain --verbose

# Upload
apio upload --project-dir=bin --board NB3_hindbrain --verbose

# FIN