SEL 0 1 2   #00 Set lower byte of reg 2 to 01
SEU 0 0 2   #01 Set upper byte of reg 2 to 00
SEL 1 0 5   #adc Set lower byte of reg 5 to 10
SEU 8 0 5   #adc Set upper byte of reg 5 to 80
STR 5 2 0   #adc (start) Store value in reg 2 at address in reg 5
SEL 0 5 3   #loop Set lower byte of reg 3 to 06
SEU 0 0 3   #loop Set upper byte of reg 3 to 00
JMP 3 0 7   #loop Jump to instruction in reg 3
#FIN