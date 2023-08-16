SEL 0 0 2   # Set lower byte of reg 2 to 00
SEU 0 0 2   # Set upper byte of reg 2 to 00
SEL 0 1 3   # Set lower byte of reg 3 to 01
SEU 0 0 3   # Set upper byte of reg 3 to 00
SEL 0 7 4   # Set lower byte of reg 2 to 07
SEU 0 0 4   # Set upper byte of reg 2 to 00
SEL 0 3 5   # Set lower byte of reg 2 to 01
SEU 0 0 5   # Set upper byte of reg 2 to 00
STR 2 4 0   # Store value in reg 4 at address in reg 2
STR 3 5 0   # Store value in reg 5 at address in reg 3
LOD 2 0 6   # Load value at address in reg 2 into reg 6
LOD 3 0 7   # Load value at address in reg 3 into reg 7
ADD 6 7 8   # Add values in regs 6 and 7, store result in reg 8
#FIN