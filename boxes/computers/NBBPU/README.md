# computers : NBBPU

The No Black Box Processing Unit

## Overview

A custom CPU loosely based on the MOS 6502 (and the RISC-V/AVR ISAs)

## Instructions

All 16-bit instructions are of the format:

```
OpCode(0:3) regX(4:7) regY(8:11) regZ(12:15)"
```

PC: 16-bit unsigned (max 65K instructions)
Data type: Signed 16-bit intergers (-32,768 to +32,767)
Registers (16x16-bit): r0 to r15

### Arithmetic and Logic

1. ADD (addition: x + y => z): **ADD x y z**
2. SUB (subtraction: x - y => z): **SUB x y z**
3. AND (logical "and": x ^ y => z): **AND x y z**
4. OR (logical "or": x | y => z): **OR x y z**
5. XOR (logical "exclusive or": x ⊕ y => z): **XOR x y z**

### Control Flow
6. BREQ (branch to z if x == y): **BREQ x y z** 
7. BRNEQ (branch to z if x != y): **BRNEQ x y z**
8. BRLT (branch to z is x < Y): **BRLT x y z**
9. BRGT (branch if *reg1* is greater than *reg2*): **BRGT x y z**
10. JUMP (jump PC to x, store nest instruction in z): **JUMP x 0 z**
11. RET (return from jump): **RET 0 0 0**

### Memory
12. LOAD (load data from memory adress z into x): **LOAD x 0 z**
13. STORE (store data in x at memory address z): **STORE x 0 z**

### Other
14. NOP (no operation): **NOP 0 0 0**
15. SETL (set lower byte of z): **SETL -byte* z**
16. SETU (set upper byte of z): **SETU -byte* z**
