SEL 0 4 6   #01 Set lower byte of reg 6 to 04 - Outer loop address (4)
SEU 0 0 6   #02 Set upper byte of reg 6 to 00 - 
SEL 0 A 4   #03 Set lower byte of reg 4 to 03 - Num Loops
SEU 0 0 4   #04 Set upper byte of reg 4 to 00 -
SEL F F 2   #05 Set lower byte of reg 2 to FF - Loop limit
SEU 7 F 2   #06 Set upper byte of reg 2 to FF - 
SEL 0 1 3   #07 Set lower byte of reg 3 to 01 - Loop step
SEU 0 0 3   #08 Set upper byte of reg 3 to 00 - 
SEL 0 A 5   #09 Set lower byte of reg 5 to 0A - Inner loop address
SEU 0 0 5   #10 Set upper byte of reg 5 to 00 - 
SUB 2 3 2   #11 Subtract values in regs 3 and 2, store result in reg 2
BRN 5 2 0   #12 Branch to reg 5 if reg 2 != 0
SUB 4 3 4   #13 Subtract values in regs 4 and 3, store result in reg 4
BRN 6 4 0   #14 Branch to reg 6 if reg 4 != 0
SEL 0 0 2   #15 Set lower byte of reg 2 to 00
SEU 0 0 2   #16 Set upper byte of reg 2 to 00
SEL 0 0 3   #17 Set lower byte of reg 3 to 01
SEU 0 0 3   #18 Set upper byte of reg 3 to 00
STR 2 3 0   #19 Store value in reg 3 at address in reg 2
JMP 0 0 1   #20 Repeat
#FIN