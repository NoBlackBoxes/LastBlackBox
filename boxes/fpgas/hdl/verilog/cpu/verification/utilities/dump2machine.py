# Should be upper case HEX?
# Clean code...


# -*- coding: utf-8 -*-
import sys

# Parse command line input
if(len(sys.argv) == 2):
    input_path = sys.argv[1]
    output_path = input_path[:-4] + 'txt'
else:
    exit("Fail: dump2machine - incorrect input args")

# Open input (dump) and output (machine code) file
input_file = open(input_path, "r")
output_file = open(output_path, "w")

# Load input HEX dump (text) file
lines = input_file.readlines()

# Parse input HEX dump file
for line in lines:
    tokens = line.split(' ')
    num_tokens = len(tokens)
    if(num_tokens > 1):
        lower = False
        code = ''
        for token in tokens[1:]:
            if len(token) >= 4:
                token = token[:4] # Remove newline
                if lower:
                    output_file.write(token + code + '\n')                    
                    lower = False
                else:
                    code = token
                    lower = True

# Cleanup
input_file.close()
output_file.close()

#FIN