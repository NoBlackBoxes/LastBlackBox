import sys
from libs.opcodes import OpCodes
from libs.system import State
from libs.operations import operation

# Define state
state = State()

# Check for program to emulate
if len(sys.argv) != 2:
    print("Usage: python emulate.py <executable.rom>")
    exit()

# Extract input file to emulate
input_path = sys.argv[1]

# Open file
input_file = open(input_path, 'r')

# Load program
instructions = input_file.readlines()

# Emulate
emulating = True
state.pc = 0
count = 0
while emulating:
    # Fetch instruction
    instruction = instructions[state.pc]

    # Seperate instrution nibbles
    op = instruction[0]
    x = int(instruction[1], 16)
    y = int(instruction[2], 16)
    z = int(instruction[3], 16)

    # Report instruction
    #print("{0:03d}: {1} {2} {3} {4}".format(state.pc, OpCodes[op], x, y, z))

    # Run operation
    operation(OpCodes[op], x, y, z, state)

    # Report registers
    #print("\t{0}".format(state.registers))

    # DEBUG
    if count > 10000000:
        print("\t{0}".format(state.registers))
        break
    else:
        count = count + 1

# Shutdown
input_file.close()

#FIN