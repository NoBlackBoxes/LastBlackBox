#!/bin/bash
set -eu

# Create out directory
mkdir -p bin

# Build
iverilog -o bin/hello hello.v

# Simulate
vvp bin/hello