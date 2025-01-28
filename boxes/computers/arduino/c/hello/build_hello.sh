#!/bin/bash
set -eu

# Create output directory
mkdir -p bin
mkdir -p bin/interim

# Compile
avr-gcc -mmcu=atmega328p -Wall -Os -DF_CPU=16000000 hello.c -o bin/interim/hello.out

# Link
avr-ld -o bin/interim/hello.elf bin/interim/hello.out

# Something?
avr-objcopy --output-target=ihex bin/interim/hello.elf bin/hello.hex

# Report binary size
avr-size --format=avr --mcu=atmega328p bin/interim/hello.elf

# Program the board
avrdude -v -p ATmega328p -c arduino -P /dev/ttyUSB0 -b 115200 -D -U flash:w:bin/hello.hex

#FIN