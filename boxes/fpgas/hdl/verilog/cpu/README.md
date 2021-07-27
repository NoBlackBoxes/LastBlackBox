# fpgas : hdl : verilog : cpu

We will use the Verilog hardware description language (HDL) to design a RISC-V cpu

## RISC-V

We will implement the minimum subset of the RISC-V specification: RV32I.

We need 32 x 32-bit registers.

We need memory for instructions and data in one 32-bit byte-addressed.

We also need a programme counter (PC).

We need to handle these instructions:

- R-type: reg to reg

  - add
  - sub
  - sll
  - slt
  - sltu
  - xor
  - srl
  - sra
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