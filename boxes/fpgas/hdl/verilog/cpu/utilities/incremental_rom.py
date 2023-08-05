# -*- coding: utf-8 -*-
import sys

# Parse command line input
if(len(sys.argv) == 2):
    output_path = sys.argv[1]
else:
    exit("Fail: incremental_rom - incorrect input arg count")

# Create an incremental HEX text file for ROM instruction memory
output_file = open(output_path, "w")
memory_size = 4096
for i in range(memory_size):
    hex_string = f"{i:#0{10}X}"
    output_file.write(hex_string[2:] + '\n')

# Cleanup
output_file.close()

#FIN