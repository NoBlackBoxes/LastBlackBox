import ast
from pprint import pprint

# read the python file as text
with open('hello.py', 'r') as source_file:
    source = source_file.read()

# make an AST
node = ast.parse(source, mode='exec')
pprint(ast.dump(node))

# make a code object
code_object = compile(node, '<string>', mode='exec')
# run the code
exec(code_object)
