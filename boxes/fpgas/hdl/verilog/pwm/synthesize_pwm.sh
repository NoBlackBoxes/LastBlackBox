#!/bin/bash
set -eu

# Create out directory
mkdir -p bin

# Copy verilog module(s)
cp ../pwm.v bin/.

# Copy APIO file
cp apio.ini bin/.

# Copy constraints file
cp NB3_hindbrain_pwm.pcf bin/.

# Verify Verilog
apio verify --project-dir=bin --board NB3_hindbrain --verbose 

# Synthesize
apio build --project-dir=bin --board NB3_hindbrain --verbose 


# FIN