#!/bin/bash
set -eu

# Create output directory
mkdir -p bin
mkdir -p bin/interim

# Assemble
avr-as -mmcu=atmega328p hello.asm -o bin/interim/hello.out

# Link
avr-ld -o bin/interim/hello.elf bin/interim/hello.out

# Something?
avr-objcopy --output-target=ihex bin/interim/hello.elf bin/hello.hex

# Program the board
avrdude -v -p ATmega328p -c arduino -P /dev/ttyUSB0 -b 115200 -D -U flash:w:bin/hello.hex
