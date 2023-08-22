SEL 0 8 2   #00 Set lower byte of reg 2 to 08
SEU 0 0 2   #01 Set upper byte of reg 2 to 00
SEL 0 1 3   #02 Set lower byte of reg 3 to 01
SEU 0 0 3   #03 Set upper byte of reg 3 to 00
SEL 0 6 4   #04 Set lower byte of reg 4 to 06
SEU 0 0 4   #05 Set upper byte of reg 4 to 00
SUB 2 3 2   #06 Subtract values (regs 2 - 3), store result in reg 2
STR 3 2 0   #07 Store value in reg 2 at address in reg 3
BRN 4 2 0   #08 Branch to reg 4 if reg 2 != 0
JMP 0 0 5   #09 Jumpt to start (store PC in reg 5)
#FIN