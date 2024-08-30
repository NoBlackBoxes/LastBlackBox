# -*- coding: utf-8 -*-
import sys

# Parse command line input
if(len(sys.argv) == 2):
    output_path = sys.argv[1]
else:
    exit("Fail: empty_ram - incorrect input arg count")

# Create empty HEX text file for data memory
output_file = open(output_path, "w")
memory_size = 256
for i in range(memory_size):
    output_file.write('0000\n')

# Cleanup
output_file.close()

#FIN