#!/bin/bash
set -eu

# Create output directory for binary files
mkdir -p bin
mkdir -p bin/interim

# Compile
gcc -c hello.c -o bin/interim/hello.o

# Link
gcc bin/interim/hello.o -o bin/hello