import pandas as pd
import sys
from collections import defaultdict
from anytree import Node, RenderTree


def call_table(table_name: str) -> dict:
    df = pd.read_csv(table_name + ".csv").stack().reset_index()
    df.columns = ['Row', 'Column', 'Value']
    df = df.dropna()
    table = defaultdict(dict)

    for i, v in df.iterrows():
        if isinstance(v['Value'], float):
            v['Value'] = int(v['Value'])

        table[int(v['Row'])][v['Column']] = v['Value']

    return dict(table)


# cfg 불러오기
def call_cfg(cfg_name: str) -> dict:
    df = pd.read_csv(cfg_name + ".txt")
    cfg = df.to_dict('index')
    cfg = {int(key): list(value['cfg'].split('->')) for key, value in cfg.items()}
    cfg = {int(key): [w.strip() for w in word] for key, word in cfg.items()}
    return cfg


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
        self.cur_index = 0 # left viable prefix 할 때, 사용할 cur
        self.cur_stack = [0]
        self.cfg_table = cfg # cfg 들이 담김
        self.state_stack = [Node("S")] # state 들의 stack
        self.error_flag = 0

    # parse
    def parse(self):
        accepted = False

        while not accepted:
            self.check_viable_prefix() # left check
            target = self.tokens[self.splitter + 1]

            try :
                next_rule = self.table[self.state][self.tokens[self.splitter+1]]
            except KeyError as e:
                print("REJECTED")
                print("Parsing error : token {0} from index {1}".format(target, self.splitter))
                print(' '.join(self.tokens))
                f.write("REJECTED\n")
                f.write("Parsing error : token {0} from index {1}\n".format(target, self.splitter))
                f.write(' '.join(self.tokens))
                f.write("\n")
                self.error_flag = 1
                break

            accepted = self.check_rule(next_rule, left=False)

    # viable_prefix check
    def check_viable_prefix(self):
        # 현재 state -> rule check해서 행동
        self.cur_stack = [0]
        self.state = 0
        if self.splitter == 0:
            return True
        else:
            self.cur_index = 0

        next_rule = self.table[self.state][self.tokens[self.cur_index]]

        while not self.check_rule(next_rule):
            if self.cur_index == self.splitter:
                return True
            self.cur_stack.append(self.state)
            next_rule = self.table[self.state][self.tokens[self.cur_index]]

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
            self.goto(rule, left)
            return False

        elif isinstance(rule, str):
            # shift
            if rule[0] == 's':
                self.shift(rule, left)
                return False
            elif rule[0] == 'r':
                self.reduce(rule)
                return False
            elif rule == 'acc':
                return True

    # shift
    def shift(self, rule, left):
        next_state = int(rule[1:])
        self.state = next_state

        if left:
            self.cur_index += 1
        else:
            self.forward()

    def goto(self, rule, left):
        self.state = int(rule)

        if left:
            self.cur_index += 1
        else:
            self.forward()

    def reduce(self, rule):
        cfg = self.cfg_table[int(rule[1:])]
        lhs, rhs = cfg
        self.state_stack.append(Node(lhs, parent=self.state_stack[0]))
        rhs_tokens = list(rhs.split())
        reduce_len = len(rhs_tokens)

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


# Argument values ignored after input file name
if __name__ == "__main__":
    try:
        if len(sys.argv) == 2:
            sequence = open(sys.argv[1], 'r').readline()

        my_parser = Parser(call_table("table"), sequence, call_cfg("cfg"))
        f = open("output.txt", 'w')
        my_parser.parse()
        if my_parser.error_flag == 0:
            for pre, fill, node in RenderTree(my_parser.state_stack[0], childiter=reversed):
                print("%s%s" % (pre, node.name))
                f.write("%s%s" % (pre, node.name))
                f.write("\n")
        print("Stored in output.txt")
        f.close()

    except (FileNotFoundError, NameError) as e:
        print("Invalid argument")
