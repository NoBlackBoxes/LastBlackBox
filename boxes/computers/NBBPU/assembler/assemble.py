import os
import sys
from libs.opcodes import OpCodes

# Check for file(s) to assemble
if len(sys.argv) != 3:
    print("Usage: python assemble.py <code.as> <output.rom>")
    exit()

# Extract input file to assemble
input_path = sys.argv[1]

# Set output path
output_path = sys.argv[2]

# Open files
input_file = open(input_path, 'r')
output_file = open(output_path, 'w')

# Assemble
assembling = True
instruction_count = 0
while assembling:
    # Read line
    line = input_file.readline()

    # Is finished? Stop.
    if line == '#FIN':
        assembling = False
        continue

    # Seperate code from comments
    (code, comments) = line.split('#')

    # Extract tokens
    [opcode, x, y, z] = code.split(' ')[:4]

    # Write binary
    output_file.write("{0}{1}{2}{3}\n".format(OpCodes[opcode], x, y, z))

    # Increment instruction count
    instruction_count = instruction_count + 1

# Pad ROM
max_instructions = 256
for i in range(max_instructions - instruction_count):
    output_file.write("0000\n")

# Shutdown
input_file.close()
output_file.close()

#FIN


