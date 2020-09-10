import ast
import click
from astmonkey import visitors, transformers
import os


@click.command()
@click.argument('infilename', type=click.Path(exists=True))
@click.argument('outfilename', type=click.Path())
def main(infilename, outfilename):
    node = ast.parse(open(infilename).read())
    node = transformers.ParentChildNodeTransformer().visit(node)
    visitor = visitors.GraphNodeVisitor()
    visitor.visit(node)
    visitor.graph.write_png(outfilename)
    os.system('open ' + outfilename)


if __name__ == "__main__":
    main()
