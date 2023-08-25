SEL 0 0 2   #00 Set lower byte of reg 2 to 00
SEU 0 0 2   #01 Set upper byte of reg 2 to 00
SEL 0 1 3   #02 Set lower byte of reg 3 to 01
SEU 0 0 3   #03 Set upper byte of reg 3 to 00
SEL 0 7 4   #04 Set lower byte of reg 4 to 07
SEU 0 6 4   #05 Set upper byte of reg 4 to 06
SEL 0 5 5   #06 Set lower byte of reg 5 to 05
SEU 0 4 5   #07 Set upper byte of reg 5 to 04
STR 2 4 0   #08 Store value in reg 4 at address in reg 2
STR 3 5 0   #09 Store value in reg 5 at address in reg 3
LOD 2 0 6   #10 Load value at address in reg 2 into reg 6
LOD 3 0 7   #11 Load value at address in reg 3 into reg 7
ADD 6 7 8   #12 Add values in regs 6 and 7, store result in reg 8
SUB 8 7 9   #13 Subtract values in regs 8 and 7, store result in reg 9
SEL 0 6 4   #14 Set lower byte of reg 4 to 07
SEU 0 6 4   #15 Set upper byte of reg 4 to 06
SUB 9 4 2   #16 Subtract values in regs 9 and 4, store result in reg 2
JMP 0 0 1   #17 Jump PC to value in reg 0 (0) and store return in reg 1
#FIN