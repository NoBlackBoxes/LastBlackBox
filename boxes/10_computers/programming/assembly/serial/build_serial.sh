#!/bin/bash
set -eu

# Create output directory
mkdir -p bin
mkdir -p bin/interim

# Assemble
avr-as -mmcu=atmega328p serial.asm -o bin/interim/serial.out

# Link
avr-ld -o bin/interim/serial.elf bin/interim/serial.out

# Something?
avr-objcopy --output-target=ihex bin/interim/serial.elf bin/serial.hex

# Program the board
avrdude -v -p ATmega328p -c arduino -P /dev/ttyUSB0 -b 115200 -D -U flash:w:bin/serial.hex
