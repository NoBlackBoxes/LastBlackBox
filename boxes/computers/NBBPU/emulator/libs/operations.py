# Operations Library

# Run operation
def operation(op, x, y, z, state):

    if op == "ADD":
        ADD(x, y, z, state)
    elif op == "SUB":
        SUB(x, y, z, state)
    elif op == "AND":
        AND(x, y, z, state)
    elif op == "IOR":
        IOR(x, y, z, state)
    elif op == "XOR":
        XOR(x, y, z, state)
    elif op == "SHR":
        SHR(x, y, z, state)
    elif op == "SHL":
        SHL(x, y, z, state)
    elif op == "CMP":
        CMP(x, y, z, state)
    elif op == "JMP":
        JMP(x, y, z, state)
    elif op == "BRZ":
        BRZ(x, y, z, state)
    elif op == "BRN":
        BRN(x, y, z, state)
    elif op == "RES":
        RES(x, y, z, state)
    elif op == "LOD":
        LOD(x, y, z, state)
    elif op == "STR":
        STR(x, y, z, state)
    elif op == "SEL":
        SEL(x, y, z, state)
    elif op == "SEU":
        SEU(x, y, z, state)
    else:
        print("Invalid OpCode: {0}".format(op))
        exit(-1)
    return

#
# Operations
#

# ADD
def ADD(x, y, z, state):
    state.registers[z] = state.registers[x] + state.registers[y]
    state.pc = state.pc + 1
    return

# SUB
def SUB(x, y, z, state):
    state.registers[z] = state.registers[x] - state.registers[y]
    state.pc = state.pc + 1
    return

# AND
def AND(x, y, z, state):
    state.registers[z] = state.registers[x] & state.registers[y]
    state.pc = state.pc + 1
    return

# IOR
def IOR(x, y, z, state):
    state.registers[z] = state.registers[x] | state.registers[y]
    state.pc = state.pc + 1
    return

# XOR
def XOR(x, y, z, state):
    state.registers[z] = state.registers[x] ^ state.registers[y]
    state.pc = state.pc + 1
    return

# SHR
def SHR(x, y, z, state):
    state.registers[z] = state.registers[x] >> state.registers[y]
    state.pc = state.pc + 1
    return

# SHL
def SHL(x, y, z, state):
    state.registers[z] = state.registers[x] << state.registers[y]
    state.pc = state.pc + 1
    return

# CMP
def CMP(x, y, z, state):
    if state.registers[x] >= state.registers[y]:
        state.registers[z] = 1
    else:
        state.registers[z] = 0
    state.pc = state.pc + 1
    return

# JMP
def JMP(x, y, z, state):
    state.registers[z] = state.pc + 1
    state.pc = state.registers[x]
    return

# BRZ
def BRZ(x, y, z, state):
    if state.registers[y] == 0:
        state.pc = state.registers[x]
    else:
        state.pc = state.pc + 1
    return

# BRN
def BRN(x, y, z, state):
    if state.registers[y] != 0:
        state.pc = state.registers[x]
    else:
        state.pc = state.pc + 1
    return

# RES
def RES(x, y, z, state):
    state.pc = state.pc + 1
    return

# LOD
def LOD(x, y, z, state):
    address = state.registers[x]
    state.registers[z] = state.ram[address]
    state.pc = state.pc + 1
    return

# STR
def STR(x, y, z, state):
    address = state.registers[x]
    state.ram[address] = state.registers[y]
    state.pc = state.pc + 1
    print("blink")
    return

# SEL
def SEL(x, y, z, state):
    high_nibble = x << 4
    low_nibble = y << 4 >> 4
    lower = high_nibble + low_nibble
    upper = state.registers[z] >> 8 << 8
    state.registers[z] = upper + lower
    state.pc = state.pc + 1
    return

# SEU
def SEU(x, y, z, state):
    high_nibble = x << 4
    low_nibble = y << 4 >> 4
    upper = (high_nibble + low_nibble) << 8
    lower = state.registers[z] << 8 >> 8
    state.registers[z] = upper + lower
    state.pc = state.pc + 1
    return
