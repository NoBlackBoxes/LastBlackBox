# computers : NBBPU

The No Black Box Processing Unit

## Overview

A custom CPU loosely based on the MOS 6502 (and the RISC-V/AVR ISAs)

## Instructions

All 16-bit instructions are of the format:

```
OpCode(0:3) regX(4:7) regY(8:11) regZ(12:15)"
```

- PC: 16-bit unsigned (max 65K instructions)
- Data type: Signed 16-bit intergers (-32,768 to +32,767)
- Registers (16x16-bit): r0 to r15
 -- r0 is always 0
 -- r1 is always the return adress from a jump

Name|OpCode|Description                          |Example  |
:--:|:----:|:-----------------------------------:|:-------:|
|***Arithmetic and Logic***                                |
ADD |0000  |*addition: x + y => z*               |ADD x y z|
SUB |0001  |*subtraction: x - y => z*            |SUB x y z|
AND |0010  |*logical "and": x & y => z*          |AND x y z|
IOR |0011  |*logical "inclusive or": x \| y => z*|IOR x y z|
XOR |0100  |*logical "exclusive or": x ^ y => z* |XOR x y z|
SHR |0101  |*shift x right by y-bits: x >> y = z*|SHR x y z|
SHL |0110  |*shift x left by y-bits: x << y = z* |SHL x y z|
CMP |0111  |*compare x to y: x >= y = z          |CMP x y z|
|***Control Flow***                                        |
JMP |1000  |*jump PC to x, store next PC in r1*  |JMP x 0 0|
BRE |1001  |*branch to z if x == y*              |BRE x y z| 
BRN |1010  |*branch to z if x != y*:             |BRN x y z|
RES |1011  |*reserved op code*                   |RES 0 0 0|
***Memory***                                               |
LOD |1100  |*load data at memory address z in x* |LOD x 0 z|
STR |1101  |*store data in x at memory address z*|STR x 0 z|
SEL |1110  |*set lower byte of z*                |SEL b8 z |
SEU |1111  |*set upper byte of z*                |SEU b8 z |
