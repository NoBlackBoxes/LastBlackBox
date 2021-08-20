# fpgas : hdl : verilog : cpu

We will use the Verilog hardware description language (HDL) to design a RISC-V cpu that implements the base (RV32I) instruction set.

## RISC-V: RV32I ISA
<hr>
<br>

Name |Type| ?|Description|
:----|:---|:-|:----------|
lb   |   I| x|Load       |
lh   |   I|  |Load       |
lw   |   I| x|Load       |
lbu  |   I| x|Load (U)   |
lhu  |   I|  |Load (U)   |
addi |   I| x|           |
slli |   I|  |           |
slti |   I|  |           |
sltiu|   I|  |           |
xori |   I|  |           |
srli |   I|  |           |
srai |   I|  |           |
ori  |   I|  |           |
andi |   I| ?|           |
auipc|   U|  |           |
sb   |   S|  |           |
sh   |   S|  |           |
sw   |   S| x|           |
add  |   R| x|           |
sub  |   R| x|           |
sll  |   R|  |           |
slt  |   R| x|           |
sltu |   R|  |           |
xor  |   R| ?|           |
srl  |   R|  |           |
sra  |   R|  |           |
or   |   R| x|           |
and  |   R| x|           |
lui  |   U| x|           |
beq  |   B| x|           |
bne  |   B| x|           |
blt  |   B| x|           |
bge  |   B| x|           |
bltu |   B| x|           |
bgeu |   B| x|           |
jalr |   I| x|           |
jal  |   J| x|           |

<hr>
<br>

Fetch, CSR?, ECALL?...should be 40 instructions...

## Other modules

1. We need 32 x 32-bit registers in a three port register file.
2. We need memory for instructions and data in one 32-bit byte-addressed.
3. We also need a programme counter (PC).

We need to handle these instructions:

- R-type: reg to reg

  - add
  - sub
  - sll ?
  - slt
  - sltu ?
  - xor ?
  - srl ?
  - sra ?
  - or
  - and

- I-type: (short) immediates and loads

  - addi
  - slti
  - sltiu
  - xori
  - ori
  - andi
  - slli
  - srli
  - srai
  - lb
  - lh
  - lw
  - lbu
  - lhu

- S-type: stores

  - sb
  - sh
  - sw

- B-type: branches

  - beq
  - bne
  - blt
  - bge
  - bltu
  - bgeu

- U-type: upper (long) immediates

- J-type: jumps