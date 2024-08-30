#!/bin/bash
set -eu

# Create output directory
mkdir -p bin

# Copy verilog top-level project and module(s)
cp and_gate.v bin/.

# Copy APIO file
cp apio.ini bin/.

# Copy constraints file
cp NB3_hindbrain_and_gate.pcf bin/.

# Verify Verilog
apio verify --project-dir=bin --board NB3_hindbrain --verbose

# Synthesize
apio build --project-dir=bin --board NB3_hindbrain --verbose

# FIN