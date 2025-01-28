#!/bin/bash
set -eu

# Compile
arduino-cli compile --fqbn arduino:avr:nano blink

# Upload
arduino-cli upload -p /dev/ttyUSB0 -v --fqbn arduino:avr:nano blink
