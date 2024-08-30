#!/bin/bash
set -eu

# Create out directory
mkdir -p bin

# Copy verilog module(s)
cp adc.v bin/.
cp modules/rx.v bin/.
cp modules/tx.v bin/.

# Copy APIO file
cp apio.ini bin/.

# Copy constraints file
cp NB3_hindbrain_adc.pcf bin/.

# Verify Verilog
apio verify --project-dir=bin --board NB3_hindbrain --verbose 

# Synthesize
apio build --project-dir=bin --board NB3_hindbrain --verbose 

# FIN