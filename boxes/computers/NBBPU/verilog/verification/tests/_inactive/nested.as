SEL 0 4 6   #04 Set lower byte of reg 5 to 06 - Outer loop address (4)
SEU 0 0 6   #05 Set upper byte of reg 5 to 00 - 
SEL 0 3 4   #04 Set lower byte of reg 4 to 03 - Num Loops
SEU 0 0 4   #05 Set upper byte of reg 4 to 00 -
SEL 1 0 2   #00 Set lower byte of reg 2 to 10 - Loop limit
SEU 0 0 2   #01 Set upper byte of reg 2 to 00 - 
SEL 0 1 3   #02 Set lower byte of reg 3 to 01 - Loop step
SEU 0 0 3   #03 Set upper byte of reg 3 to 00 - 
SEL 0 A 5   #04 Set lower byte of reg 5 to 06 - Inner loop address
SEU 0 0 5   #05 Set upper byte of reg 5 to 00 - 
SUB 2 3 2   #06 Subtract values in regs 3 and 2, store result in reg 2
BRN 5 2 0   #07 Branch to reg 5 if reg 2 != 0
SUB 4 3 4   #06 Subtract values in regs 4 and 3, store result in reg 4
BRN 6 4 0   #07 Branch to reg 6 if reg 4 != 0
SEL 0 0 2   #yy Set lower byte of reg 2 to 00
SEU 0 0 2   #yy Set upper byte of reg 2 to 00
SEL 0 0 3   #yy Set lower byte of reg 3 to 00
SEU F 1 3   #yy Set upper byte of reg 3 to F0
STR 2 3 0   #yy Store value in reg 3 at address in reg 2
SEL 1 7 6   #04 Set lower byte of reg 5 to 06 - Outer loop address (23)
SEU 0 0 6   #05 Set upper byte of reg 5 to 00 - 
SEL 0 3 4   #04 Set lower byte of reg 4 to 03 - Num Loops
SEU 0 0 4   #05 Set upper byte of reg 4 to 00 -
SEL 1 0 2   #00 Set lower byte of reg 2 to 10 - Loop limit
SEU 0 0 2   #01 Set upper byte of reg 2 to 00 - 
SEL 0 1 3   #02 Set lower byte of reg 3 to 01 - Loop step
SEU 0 0 3   #03 Set upper byte of reg 3 to 00 - 
SEL 1 D 5   #04 Set lower byte of reg 5 to 06 - Inner loop address (29)
SEU 0 0 5   #05 Set upper byte of reg 5 to 00 - 
SUB 2 3 2   #06 Subtract values in regs 3 and 2, store result in reg 2
BRN 5 2 0   #07 Branch to reg 5 if reg 2 != 0
SUB 4 3 4   #06 Subtract values in regs 4 and 3, store result in reg 4
BRN 6 4 0   #07 Branch to reg 6 if reg 4 != 0
SEL 0 0 2   #yy Set lower byte of reg 2 to 00
SEU 0 0 2   #yy Set upper byte of reg 2 to 00
SEL 0 0 3   #yy Set lower byte of reg 3 to 00
SEU F 0 3   #yy Set upper byte of reg 3 to F0
STR 2 3 0   #yy Store value in reg 3 at address in reg 2
SEL F 0 2   #xx Set lower byte of reg 2 to F0
SEU F F 2   #xx Set upper byte of reg 2 to FF
SEL 0 1 3   #xx Set lower byte of reg 3 to 01
SEU 0 0 3   #xx Set upper byte of reg 3 to 00
STR 2 3 0   #xx Store value in reg 3 at address in reg 2
#FIN