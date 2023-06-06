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
        self.cur_stack = []

    # parse
    def parse(self):
        print(self.table)
        self.cur_stack.append(0)
        while True:
            next_input = self.tokens[self.cur]
            next_rule = self.table[self.state][next_input]
            print("cursor:", self.cur, " next_input:", next_input)
            print("tokens:", self.tokens)
            print("Let's parse. state:", self.state, "cfg:", next_rule)
            if not self.check_rule(next_rule):
                break
            print("after. cursor:", self.cur)
            print("tokens:", self.tokens)
            print("state:", self.state)
            print("------------------------------")

    # forward / 앞으로 나가기
    def forward(self):
        self.cur += 1

    # table 속 rule 확인
    def check_rule(self, rule):
        if isinstance(rule, int):
            # goto
            print("goto")
            self.state = int(rule)
            self.cur_stack.append(self.state)
            return True

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
        self.cur_stack.append(self.state)
        print("cur_stack:", self.cur_stack)
        self.forward()

        # error_check
        pass

    def reduce(self, rule):
        # reduce
        cfg = self.cfg_table[int(rule[1:])]
        print("reduce rule: ", cfg)
        lhs, rhs = cfg
        rhs_tokens = list(rhs.split())
        reduce_len = len(rhs_tokens)
        if rhs_tokens == ["''"]:
            self.tokens.insert(self.cur, lhs)
            self.forward()

        else:
            if reduce_len == 1:
                self.tokens[self.cur-1] = lhs
            elif reduce_len > 1:
                self.tokens[self.cur-1] = lhs
                del self.tokens[(self.cur-1)-(reduce_len-1):(self.cur-1)]
                self.cur -= (reduce_len-1)

            for _ in range(reduce_len):
                self.cur_stack.pop()

            self.state = self.cur_stack[-1]
        print("cur_stack:", self.cur_stack)


        # cur로부터 len만큼 앞에 없앰 (만약 0이다?, 그냥 자기만 바꿈.) (1이상? 자기 포함 앞에 바꿈)
        # (cur바꿈)
        # (cur 바꿈) (cur - reduce_len:cur까지 del해버림)
        # cur을 cur - reduce_len으로 초기화

        # del my_list[start:end]
        # cur input이 그대로 들어감
        # lhs -> rhs
        # rhs를 다시 lhs로 바꿔야 함
        # rhs가 여러개라면 어떻게...?

        # error_check
        pass


sequence = "vtype id lparen vtype id comma vtype id rparen lbrace return num semi rbrace"
my_parser = Parser(call_table("table"), sequence, call_cfg("cfg"))
print(my_parser.cfg_table)
my_parser.parse()
