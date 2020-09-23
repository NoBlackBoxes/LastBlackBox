#!/bin/bash
set -eu

# Program the board
avrdude -v -p ATmega328p -c arduino -P /dev/ttyUSB0 -b 115200 -D -U flash:w:program.hex
