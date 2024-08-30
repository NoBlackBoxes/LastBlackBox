# Keyword (reserved) dictionary
Keywords = {
    "void"          : "VOID",
    "int"           : "INT",
    "for"           : "FOR",
    "return"        : "RETURN",
}

# Single Operator dictionary
Single_Operators = {
    '='             : "ASSIGN",
    '+'             : "PLUS",
    '-'             : "MINUS",
    '!'             : "NOT",
    '<'             : "LESS",
    '>'             : "GREATER",
}

# Double Operator dictionary
Double_Operators = {
    '--'            : "DECREMENT",
    '++'            : "INCREMENT",
    '+='            : "ASSIGN_PLUS",
    '-='            : "ASSIGN_MINUS",
    '=='            : "EQUAL",
    '!='            : "NOTEQUAL",
}

# Single Seperator dictionary
Single_Seperators = {
    "("             : "LEFT_PARENT",
    ")"             : "RIGHT_PARENT",
    "["             : "LEFT_BRACKET",
    "]"             : "RIGHT_BRACKET",
    "{"             : "LEFT_BRACE",
    "}"             : "RIGHT_BRACE",
    ","             : "COMMA",
    ";"             : "SEMICOLON",
}

# Double Seperator dictionary
Double_Seperators = {
    "/*"            : "LEFT_COMMENT",
    "*/"            : "RIGHT_COMMENT",
    "//"            : "LINE_COMMENT",
}

# Whitspace dictionary
Whitespaces = {
    ' '             : "SPACE",
    '\t'            : "TAB",
    '\n'            : "NEWLINE",
}

# Sets
Singles = Single_Seperators | Single_Operators
Doubles = Double_Seperators | Double_Operators

# Letters
lower_case = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
upper_case = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
_Letters = ['_'] + lower_case + upper_case

# Digits
Digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# Grammar for NBB "C" language
# ----------------------------
#
# Program
# └── Declarations
#     ├── Declaration
#     │   ├── "int ID"
#     │   │   └── Declaration_Type
#     │   │       ├── Variable_Declaration
#     │   │       │   └── ";"    
#     │   │       └── Function_Declaration
#     │   │           └── "(" Parameters ")" StatementBlock
#     │   └── "void ID"
#     │           └── Function_Declaration
#     │               └── "(" Parameters ") StatementBlock
#     └── Declarations, Declaration
#
# Parameters
# ├── ParameterList
# │   ├── Parameter
# │   │   └── "int ID"
# │   └── ParameterList, Parameter
# └── void

# Production dictionary
Productions = {
    'Program'               : [(['Declarations']                            , 'Program: Declarations')],
    'Declarations'          : [(['Declaration']                             , 'Declarations: -single-'),
                               (['Declarations', 'Declaration']             , 'Declarations: -multiple-')],
}

#    'Declaration'           : (['int', 'ID', 'Declaration_Type']            , print('Declaration: int ID Declation_Type')),
#    'Declaration'           : (['void', 'ID', 'Function_Declaration']       , print('Declaration: void ID Function_Declaration')),
#    'Declaration_Type'      : (['Variable_Declaration']                     , print('Declaration_Type: Variable_Declaration')),
#    'Declaration_Type'      : (['Function_Declaration']                     , print('Declaration_Type: Function_Declaration')),
#    'Variable_Declaration'  : ([';']                                        , print('Variable_Declaration: ;')),
#    'Function_Declaration'  : (['(', 'Parameters', ')', 'StatementBlock']   , print('Function_Declaration: (Parameters) StatementBlock')),
#    'Parameters'            : (['ParameterList']                            , print('Parameters: ParameterList')),
#    'Parameters'            : (['void']                                     , print('Parameters: void')),
#}
#
#    'ParameterList', ['Parameter']],
#    'ParameterList', ['ParameterList', ',', 'Parameter']],
#    'Parameter', ['int', 'ID']],
#    'StatementBlock', ['{', 'InternalDeclaration',  'StatementString', '}']],
#    'InternalDeclaration', ['NULL']],
#    'InternalDeclaration', ['InternalDeclaration',  'InternalVariableDeclaration', ';']],      # change
#    'InternalVariableDeclaration', ['int', 'ID']],
#    'InternalVariableDeclaration', ['double', 'ID']],     # add
#    'StatementString', ['Statement']],
#    'StatementString', ['StatementString', 'Statement']],
#    'Statement', ['ifStatement']],
#    'Statement', ['whileStatement']],
#    'Statement', ['returnStatement']],
#    'Statement', ['AssignmentStatement']],
#    'AssignmentStatement', ['ID', '=', 'Expression', ';']],
#    'returnStatement', ['return', 'ReturnValue']],
#    'ReturnValue', ['Expression', ';']],     # change
#    'ReturnValue', [';']],             # change
#    'whileStatement', ['while', '(', 'Expression', ')', 'StatementBlock']],
#    'ifStatement', ['if', '(', 'Expression', ')', 'StatementBlock', 'elseStatement']],
#    'elseStatement', ['else', 'StatementBlock']],
#    'elseStatement', ['NULL']],
#    'Expression', ['AdditiveExpression']],
#    'Expression', ['Expression', 'relop', 'AdditiveExpression']],
#    'relop', ['<']],
#    'relop', ['<=']],
#    'relop', ['>']],
#    'relop', ['>=']],
#    'relop', ['==']],
#    'relop', ['!=']],
#    'AdditiveExpression', ['Term']],
#    'AdditiveExpression', ['AdditiveExpression', '+', 'Term']],
#    'AdditiveExpression', ['AdditiveExpression', '-', 'Term']],
#    'Term', ['Factor']],
#    'Term', ['Term', '*', 'Factor']],
#    'Term', ['Term', '/', 'Factor']],
#    'Factor', ['num']],
#    'Factor', ['(', 'Expression', ')']],
#    'Factor', ['ID', 'FTYPE']],
#    'FTYPE', ['call']],
#    'FTYPE', ['NULL']],
#    'call', ['(', 'ActualParameterList', ')']],
#    'ActualParameter', ['ActualParameterList']],
#    'ActualParameter', ['NULL']],
#    'ActualParameterList', ['Expression']],
#    'ActualParameterList', ['ActualParameterList', ',', 'Expression']],
#    'ID', ['Identifier']]

