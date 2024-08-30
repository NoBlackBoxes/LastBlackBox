# TO DO:
# - IMPLEMENT all statements (for, while, if, return, etc.)
# - IMPLEMENT expressions ((a + 2), !b, etc.))
# - Boolean Expressions?
# - Expressions!

from libs.language import *
from libs.token import Token

# Parser Class
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current = None
        self.next()

    def next(self):
        self.current = self.tokens.pop(0) if self.tokens else None
        if self.current.type == 'COMMENT':
            self.next()
        return

    def match(self, type):
        if self.current.type == type:
            self.next()
        else:
            print(f"Syntax Error: Unexpected token type ({self.current}, expected: {type}) at {self.current.line}")
            exit(-1)
        return

    def generate(self, code):
        print(code)
        return

    def error(self, message):
        print(f"\n{message} at {self.current.line}")
        print(f" - Token: {self.current}")
        exit(-1)

    def parse(self):
        self.program()
        return

    # Semantic Analysis
    # ------------------

    # Program
    def program(self):
        while self.current is not None:
            self.declaration()
        return

    # Declaration
    def declaration(self):
        if self.current.type in ['INT', 'VOID']:
            self.match(self.current.type)
            identifier = self.current.value
            self.match('ID')
            if self.current.type == 'LEFT_PARENT':
                self.function_declaration(identifier)
            else:
                self.variable_declaration(identifier)
        else:
            self.error("Syntax Error: Invalid Declaration")
        return
            
    # Variable Declaration
    def variable_declaration(self, identifier):
        if self.current.type == 'ASSIGN':
            self.match('ASSIGN')
            if self.current.type == 'INTEGER':
                self.generate(f"Reserve {identifier}")                
                self.generate(f"{identifier} = {self.current.value}")                
                self.match('INTEGER')
            elif self.current.type == 'ID':
                self.generate(f"Reserve {identifier}")                
                self.generate(f"{identifier} = {self.current.type}")                
                self.match('ID')
            self.match('SEMICOLON')
        else:
            self.generate(f"Reserve {identifier}")                
            self.match('SEMICOLON')
        return

    # Function Declaration
    def function_declaration(self, identifier):
        self.match('LEFT_PARENT')
        self.generate(f"function ({identifier})")
        self.parameters()
        self.match('RIGHT_PARENT')
        self.match('LEFT_BRACE')
        self.block()
        self.match('RIGHT_BRACE')
        return
    
    # Parameters
    def parameters(self):
        if self.current.type != 'RIGHT_PARENT':
            self.match('INT')
            identifier = self.current.value
            self.match('ID')
            self.generate(f"LOCAL(Reserve {identifier})")
            if self.current.type == 'COMMA':
                self.match('COMMA')
                self.parameters()
        return

    # Block
    def block(self):
        while self.current.type != 'RIGHT_BRACE':
            self.statement()
        return

    # Statement
    def statement(self):
        if self.current.type == 'INT':
            self.match('INT')
            identifier = self.current.value
            self.match('ID')
            self.variable_declaration(identifier)
        elif self.current.type == 'FOR':
            self.match('FOR')
            self.generate(f"For Loop...")
            self.for_loop()
        else:
            self.error("Syntax Error: Invalid Declaration")
        return

    # For Loop
    def for_loop(self):
        self.match('LEFT_PARENT')
        # Intializer
        if self.current.type == 'ID':
            identifier = self.current.value
            self.match('ID')
            self.match('ASSIGN')
            self.generate(f"Assign {identifier} = {self.current.value}")
            self.match('INTEGER')
            self.match('SEMICOLON')
        elif self.current.type == 'INT':
            self.match('INT')
            identifier = self.current.value
            self.match('ID')
            self.variable_declaration(identifier)
        else:
            self.error(f"Syntax Error: For Loop - expected an intializer assignment or declaration")
        # Condition
        self.boolean_expression()
        return

    # Boolean Expression
    def boolean_expression(self):
        # Parse boolean expression
        self.generate(f"...parsing Boolean expression")
        identifier = self.current.value
        self.match('ID')
        if self.current.type == 'LESS':
            self.generate(f"Less Than comparison")
            self.match('LESS')
        return

#FIN