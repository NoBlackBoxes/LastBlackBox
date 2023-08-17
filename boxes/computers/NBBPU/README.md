# computers : NBBPU

The No Black Box Processing Unit

## Overview

A custom CPU loosely based on the MOS 6502 (and the RISC-V/AVR ISAs)

## Requirements

```bash
# Create a python virtual environment
mkdir _tmp
cd _tmp
python -m venv nbbpu
source nbbpu/bin/activate

# Install python packages
pip install apio

# Install APIO packages
apio install system scons ice40 iverilog  # or just oss-cad-suite
apio drivers --serial-enable

# Add user to dialout (not sure for Upduino 3.0)
sudo usermod -a -G dialout $USER # Debian
sudo usermod -a -G uucp $USER # Arch
```

## Instructions

All 16-bit instructions are of the format:

```
OpCode(0:3) regX(4:7) regY(8:11) regZ(12:15)
```

- PC: 16-bit unsigned (max 65K instructions)
- Data type: Signed 16-bit intergers (-32,768 to +32,767)
- Registers (16x16-bit): r0 to r15
 -- r0 is always 0
 -- r1 *should* always be the return adress from a jump

Name|OpCode|Description                          |Example  |
:--:|:----:|:-----------------------------------:|:-------:|
|***Arithmetic and Logic***                                |
ADD |0:0000|*addition: x + y => z*               |ADD x y z|
SUB |1:0001|*subtraction: x - y => z*            |SUB x y z|
AND |2:0010|*logical "and": x & y => z*          |AND x y z|
IOR |3:0011|*logical "inclusive or": x \| y => z*|IOR x y z|
XOR |4:0100|*logical "exclusive or": x ^ y => z* |XOR x y z|
SHR |5:0101|*shift x right by y-bits: x >> y = z*|SHR x y z|
SHL |6:0110|*shift x left by y-bits: x << y = z* |SHL x y z|
CMP |7:0111|*compare x to y: x >= y = z*         |CMP x y z|
|***Control Flow***                                        |
JMP |8:1000|*jump PC to x, store next PC in z*   |JMP x 0 z|
BRZ |9:1001|*branch to x if y == 0*              |BRZ x y 0| 
BRN |A:1010|*branch to x if y != 0*:             |BRN x y 0|
RES |B:1011|*reserved op code*                   |RES 0 0 0|
***Memory***                                               |
LOD |C:1100|*load data at address x into z*      |LOD x 0 z|
STR |D:1101|*store data in y at address x*       |STR x y 0|
SEL |E:1110|*set lower byte of z*                |SEL b8 z |
SEU |F:1111|*set upper byte of z*                |SEU b8 z |
