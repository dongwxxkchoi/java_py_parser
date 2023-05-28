class Tokenizer(object):
    def __init__(self, sequence):
        self.sequence = sequence
        self.tokens = list()

    def tokenize(self):
        self.tokens = list(self.sequence.split())
        return self.tokens
