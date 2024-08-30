from libs.language import *
from libs.token import Token

# Lexer FSM States
LEXER_SEEK = 0
LEXER_TOKEN = 1
LEXER_LINE_COMMENT = 2
LEXER_BLOCK_COMMENT = 3

# Lexer Class
class Lexer:
    def __init__(self, characters):
        self.characters = characters
        self.num_characters = len(characters)
        self.tokens = []

        # Build "line number" array for debugging
        self.lines = []
        count = 1
        for c in self.characters:
            self.lines.append(count)
            if c == '\n':
                count += 1
        
        # Append '#' character to avoid out-of-bounds
        self.characters += '#'

    def tokenize(self):
        lexeme = ''
        index = 0
        state = LEXER_SEEK

        # Lexer FSM
        while(True):

            # Check for end of characters
            if index >= self.num_characters:
                break

            # Get current character and next doublet
            c = self.characters[index]
            cc = c + self.characters[index + 1]

            # LEXER_SEEK State
            if LEXER_SEEK == state:
                if c in Whitespaces:
                    index += 1
                else:
                    state = LEXER_TOKEN

            # LEXER_TOKEN State
            elif LEXER_TOKEN == state:
                # If lexeme is complete, then tokenize it...
                if (c in Whitespaces | Singles ) or (cc in Doubles):
                    if lexeme != '':
                        # Tokenize lexeme (Keyword, Number, or Identifier?)
                        if lexeme in Keywords:
                            self.tokens.append(Token(Keywords[lexeme], lexeme, self.lines[index]))
                        elif (lexeme[0] in Digits) or (lexeme[0] == '-'):
                            for l in lexeme[1:]:
                                if l in Digits:
                                    continue
                                else:
                                    print(f"Lexer Error: Invalid ID ({lexeme} at Line: {self.lines[index]})")
                                    exit(-1)
                            self.tokens.append(Token('INTEGER', int(lexeme), self.lines[index]))
                        else:
                            self.tokens.append(Token('ID', lexeme, self.lines[index]))
                    lexeme = ''
                    state = LEXER_SEEK
                # ...otherwise concat character and continue
                else:
                    lexeme += c
                    index += 1
                    continue
                # Handle Seperators and Operators
                if c in Single_Seperators and cc not in Double_Seperators:
                    self.tokens.append(Token(Single_Seperators[c], c, self.lines[index]))
                    index += 1
                    state = LEXER_SEEK
                elif c in Single_Operators and cc not in Double_Operators:
                    if (c == '-') and (cc[1] in Digits):
                        lexeme += c
                        index += 1
                        continue                 
                    else:
                        self.tokens.append(Token(Single_Operators[c], c, self.lines[index]))
                        index += 1
                        state = LEXER_SEEK
                elif cc in Double_Seperators:
                    if cc == '//':
                        lexeme = ''
                        index += 2
                        state = LEXER_LINE_COMMENT
                    elif cc == '/*':
                        lexeme = ''
                        index += 2
                        state = LEXER_BLOCK_COMMENT
                    else:
                        lexeme = ''                   
                elif cc in Double_Operators:
                        self.tokens.append(Token(Double_Operators[cc], cc, self.lines[index]))
                        index += 2
                        state = LEXER_SEEK

            # LEXER_LINE_COMMENT State
            elif LEXER_LINE_COMMENT == state:
                if c == '\n':
                    self.tokens.append(Token('COMMENT', lexeme, self.lines[index]))
                    lexeme = ''
                    index += 1
                    state = LEXER_SEEK
                else:
                    lexeme += c
                    index += 1

            # LEXER_BLOCK_COMMENT State
            elif LEXER_BLOCK_COMMENT == state:
                if cc == '*/':
                    self.tokens.append(Token('COMMENT', lexeme, self.lines[index]))
                    lexeme = ''
                    index += 2
                    state = LEXER_SEEK
                else:
                    lexeme += c
                    index += 1

        return self.tokens

    def save(self, path):
        f = open(path, 'w+')
        for t in self.tokens:
            print(t)
            print(t,  file=f)
        f.close()
        return