import ast
from astmonkey import visitors, transformers
import os

print_function = ast.Name(
    id='print',
    ctx=ast.Load(),
    lineno=1,
    col_offset=0,
)
print_args = [ast.Str(
    s=u'Hello',
    lineno=1,
    col_offset=0,
)]
function_call = ast.Call(
    keywords=[], lineno=1, col_offset=0, func=print_function, args=print_args)
expression = ast.Expr(lineno=1, col_offset=0, value=function_call)
module = ast.Module(lineno=1, col_offset=0, body=[expression])

node = transformers.ParentChildNodeTransformer().visit(module)
visitor = visitors.GraphNodeVisitor()
visitor.visit(node)
visitor.graph.write_png('handmade_ast.png')
os.system("open handmade_ast.png")

code = compile(module, '', 'exec')
print(code)
print(type(code))
exec(code)
