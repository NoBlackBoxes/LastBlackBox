import parser
from pprint import pprint

with open('hello.py', 'r') as r:
    # this makes a parse tree
    st = parser.suite(r.read())
st_list = parser.st2list(st)
pprint(st)
# pprint(st_list)
code = st.compile()
eval(code)
