# Token Class
class Token:
    def __init__(self, type, value, line):
        self.type = type
        self.value = value
        self.line = line

    def __repr__(self):
        if self.value is not None:
            return f"{self.line:04d}:Token({self.type}, {repr(self.value)})"
        else:
            return f"{self.line:04d}:Token({self.type})"
