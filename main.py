import pandas as pd
import math
from collections import defaultdict


def call_table(table_name: str) -> dict:
    df = pd.read_excel(table_name + ".xlsx").stack().reset_index()
    df.columns = ['Row', 'Column', 'Value']
    df = df.dropna()

    table = defaultdict(dict)
    for i, v in df.iterrows():
        if isinstance(v['Value'], float):
            v['Value'] = int(v['Value'])

        table[int(v['Row'])][v['Column']] = v['Value']

    return table


# cfg 불러오기
def call_cfg(cfg_name: str) -> dict:
    df = pd.read_csv(cfg_name + ".txt")
    cfg = df.to_dict('index')
    cfg = {int(key): list(value['cfg'].split('->')) for key, value in cfg.items()}
    cfg = {int(key): [w.strip() for w in word] for key, word in cfg.items()}
    return cfg


# code 받기
def get_code(code: str):
    pass


# tokenize
def tokenizer(sequence: str):
    tokens = list(sequence.split())
    return tokens

# Parser
class Parser(object):
    def __init__(self, table, tokens, cfg):
        self.table = table
        self.tokens = tokenizer(tokens)
        self.cur = 0
        self.state = 0
        self.cfg_table = cfg

    # parse
    def parse(self):
        print(self.table)
        while True:
            next_input = self.tokens[self.cur]
            next_rule = self.table[self.state][next_input]
            print(self.cur, self.state, next_input, next_rule)
            if not self.check_rule(next_rule):
                break

    # forward / 앞으로 나가기
    def forward(self):
        self.cur += 1

    # table 속 rule 확인
    def check_rule(self, rule):
        if isinstance(rule, int):
            # goto
            self.state = int(rule)

        elif isinstance(rule, str):
            # shift
            if rule[0] == 's':
                self.shift(rule)
                return True
            elif rule[0] == 'r':
                self.reduce(rule)
                return True
            elif rule == 'acc':
                return False
            else: # error
                return False

    def shift(self, rule):
        # shift
        self.state = int(rule[1:])
        self.forward()

        # error_check
        pass

    def reduce(self, rule):
        # reduce
        cfg = self.cfg_table[int(rule[1:])]
        lhs, rhs = cfg
        # lhs -> rhs
        # rhs를 다시 lhs로 바꿔야 함
        # rhs가 여러개라면 어떻게...?

        # error_check
        pass


sequence = "vtype id semi vtype id lparen rparen lbrace if lparen boolstr comp boolstr rparen lbrace rbrace"
my_parser = Parser(call_table("table"), sequence, call_cfg("cfg"))
my_parser.parse()
