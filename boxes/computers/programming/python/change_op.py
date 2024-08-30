import ast


class ReplaceBinOp(ast.NodeTransformer):
    """Replace operation by addition in binary operation"""

    def visit_BinOp(self, node):
        return ast.BinOp(left=node.left, op=ast.Add(), right=node.right)


x = 0
print("before new code x =", x)

tree = ast.parse("x = 1/3")
ast.fix_missing_locations(tree)
print(ast.dump(tree))
exec(compile(tree, 'test.py', 'exec'))
print(x)

print("after new code x =", x)

tree = ReplaceBinOp().visit(tree)
ast.fix_missing_locations(tree)
print(ast.dump(tree))
exec(compile(tree, 'test.py', 'exec'))

print("after op change x =", x)
