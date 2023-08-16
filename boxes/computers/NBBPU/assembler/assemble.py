import os
import sys

# OpCode dictionary
OpCodes = {
    "ADD" : "0",
    "SUB" : "1",
    "AND" : "2",
    "IOR" : "3",
    "XOR" : "4",
    "SHR" : "5",
    "SHL" : "6",
    "CMP" : "7",
    "JMP" : "8",
    "BRE" : "9",
    "BRN" : "A",
    "RES" : "B",
    "LOD" : "C",
    "STR" : "D",
    "SEL" : "E",
    "SEU" : "F",
}

# Check for file(s) to assemble
if len(sys.argv) != 2:
    print("Usage: python assemble.py <code.as>")
    exit()

# Extract input file to assemble
input_path = sys.argv[1]

# Set output path
output_path = os.path.basename(input_path).split('.')[0] + '.rom'

# Open files
input_file = open(input_path, 'r')
output_file = open(output_path, 'w')

# Assemble
assembling = True
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

# Shutdown
input_file.close()
output_file.close()

#FIN


