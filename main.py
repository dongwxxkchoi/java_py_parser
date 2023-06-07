import pandas as pd
import math
import sys
from collections import defaultdict
from anytree import Node, RenderTree
from anytree.exporter import DotExporter


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
    tokens = ["|"] + list(sequence.split()) + ["$"]
    return tokens


# Parser
class Parser(object):
    def __init__(self, table, tokens, cfg):
        self.table = table # parsing table
        self.tokens = tokenizer(tokens) # tokens list
        self.splitter = 0 # splitter |
        self.state = 0 # 이동할 state
        self.cur = 0 # left viable prefix 할 때, 사용할 cur
        self.cur_stack = [0]
        self.cfg_table = cfg # cfg 들이 담김
        self.state_stack = [Node("S")] # state 들의 stack

    # parse
    def parse(self):
        # print(self.table)
        while True:
            # print("\n===============================================================================================\n")
            # print("-----------------------start left viable prefix check-----------------------")
            if self.check_viable_prefix(): # left check
                target = self.tokens[self.splitter + 1]
                next_rule = self.table[self.state][target]
            else:
                print("error")
            # print("-----------------------left viable prefix check finish-----------------------")

            next_rule = self.table[self.state][self.tokens[self.splitter+1]]
            # print()
            # print("-----------------------parsing start-----------------------")
            # print("Parsing rule:", next_rule)
            if not self.check_rule(next_rule, left=False): # 실제 parse
                break
            # print("After parsing:", self.tokens)
            # print("-----------------------parsing finish-----------------------")


    # viable_prefix check
    def check_viable_prefix(self):
        # 현재 state -> rule check해서 행동
        self.cur_stack = [0]
        self.state = 0
        if self.splitter == 0:
            return True
        else:
            self.cur = 0
        next_rule = self.table[self.state][self.tokens[self.cur]]
        # print("start cursor:", self.cur, "start state:", self.state, "start rule:", next_rule)
        while self.check_rule(next_rule):
            if self.cur == self.splitter: # viable prefix가 handle 맞음
                # print("final state: ", self.state)
                return True
            self.cur_stack.append(self.state)
            next_rule = self.table[self.state][self.tokens[self.cur]]
            # print("current cursor:", self.cur, "current state:", self.state, "current rule:", next_rule)

        # print("\n\nViable prefix failed!")
        return False



    # forward / 앞으로 나가기
    def forward(self):
        self.tokens[self.splitter], self.tokens[self.splitter + 1] \
            = self.tokens[self.splitter + 1], self.tokens[self.splitter]
        self.state_stack.append(Node(self.tokens[self.splitter],
                                     parent=self.state_stack[0]))
        self.splitter += 1


    # table 속 rule 확인
    def check_rule(self, rule, left=True):
        if isinstance(rule, int):
            # goto
            self.state = int(rule)
            if left:
                self.cur += 1
            else:
                # print("rule:", rule)
                self.forward()
            return True

        elif isinstance(rule, str):
            # shift
            if rule[0] == 's':
                self.shift(rule, left)
                return True
            elif rule[0] == 'r':
                self.reduce(rule, left)
                return True
            elif rule == 'acc':
                # print("-----------------------parsing finish-----------------------")
                # print("\n\nPROGRAM FINISHED!")
                return False
            else: # exception
                return False

    # shift
    def shift(self, rule, left):
        # 전처리
        next_state = int(rule[1:])
        # action
        # state shift
        self.state = next_state
        if left:
            self.cur += 1
        else:
            self.forward()
        # exception
        pass

    def reduce(self, rule, left):
        # reduce
        if left:
            return False

        cfg = self.cfg_table[int(rule[1:])]
        # print("reduce rule: ", cfg)
        lhs, rhs = cfg
        self.state_stack.append(Node(lhs, parent=self.state_stack[0]))
        rhs_tokens = list(rhs.split())
        reduce_len = len(rhs_tokens)

        # epsilon reduce
        if rhs_tokens == ["''"]:
            self.state_stack.append(Node("ε",
                                         parent=self.state_stack[-1]))
            self.tokens.insert(self.splitter, lhs)
            self.splitter += 1

        else:
            if reduce_len == 1:
                self.tokens[self.splitter-1] = lhs
            elif reduce_len > 1:
                self.tokens[self.splitter-1] = lhs
                del self.tokens[(self.splitter-1)-(reduce_len-1):(self.splitter-1)]
                self.splitter -= (reduce_len-1)
            tmp = 0
            for i in range(len(self.state_stack) - 2, -1, -1):
                if self.state_stack[i].name in rhs_tokens and \
                        self.state_stack[i].parent == self.state_stack[0]:
                    self.state_stack[i].parent = self.state_stack[-1]
                    tmp += 1
                if tmp == len(rhs_tokens):
                    break
            for _ in range(reduce_len):
                self.cur_stack.pop()

        pass

# Argument values ignored after input file name
try:
    sequence = "vtype id lparen vtype id comma vtype id rparen lbrace return num semi rbrace"
    # sequence = "class id lbrace vtype id assign num semi vtype id lparen vtype id comma " \
    #            "vtype id rparen lbrace return num semi rbrace rbrace"

    # sequence = "vtype id assign lparen num addsub num rparen multdiv num semi vtype id lparen vtype " \
    #            "id rparen lbrace if lparen boolstr comp boolstr comp boolstr rparen lbrace id assign " \
    #            "num semi rbrace return num semi rbrace"

    if len(sys.argv) == 2:
        sequence = open(sys.argv[1], 'r').readline()

    my_parser = Parser(call_table("table"), sequence, call_cfg("cfg"))
    # print(my_parser.cfg_table)
    my_parser.parse()
    for pre, fill, node in RenderTree(my_parser.state_stack[0], childiter=reversed):
        print("%s%s" % (pre, node.name))

except (FileNotFoundError, NameError) as e:
    print("Invalid argument")
