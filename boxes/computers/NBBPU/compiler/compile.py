import sys
from libs.token import Token
from libs.lexer import Lexer
from libs.parser import Parser

# Check for a program to compile
if len(sys.argv) != 2:
    print("Usage: python compile.py <program.nbb>")
    exit()

# Extract program path/name
program_path = sys.argv[1]
program_name = program_path[:-5]

# Load program
with open(program_path) as f:
    program = f.read()

# Lexical analysis
print("Lexing...")
lexer = Lexer(program)
tokens = lexer.tokenize()
lexer.save(program_name + ".tokens")

# Syntax Parser
print("Parsing...")
parser = Parser(tokens)
result = parser.parse()
print(result)

# Debug
#print(program)
#print(tokens)

