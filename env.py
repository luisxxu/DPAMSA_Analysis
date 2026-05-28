import platform
import numpy as np
# Improvement 1: collections.deque replaces plain list for not_aligned.
# deque.popleft() is O(1) vs list.pop(0) which is O(n) because every element
# must be shifted left after removal. For long sequences this was a significant
# per-step bottleneck.
from collections import deque
import copy
import config
from itertools import combinations

colors = ["#FFFFFF", "#5CB85C", "#5BC0DE", "#F0AD4E", "#D9534F", "#808080"]

# Embedding improvement A: each IUPAC ambiguity code now has its own unique token
# ID instead of being collapsed onto a canonical base.  The previous mapping
# lost information — e.g. R (purine: A or G) was treated identically to A.
# IDs: pad=0, A=1, T=2, C=3, G=4, gap=5, N=6, R=7, W=8, K=9, Y=10
nucleotides_map = {
    # Canonical bases (upper and lower case)
    'A': 1, 'a': 1,
    'T': 2, 't': 2,
    'C': 3, 'c': 3,
    'G': 4, 'g': 4,
    # Gap character
    '-': 5,
    # IUPAC ambiguous codes — now distinct token IDs (previously all mapped to A or T)
    'N': 6,  'n': 6,   # any nucleotide (A, T, C, G)
    'R': 7,  'r': 7,   # purine        (A, G)
    'W': 8,  'w': 8,   # weak bond     (A, T)
    'K': 9,  'k': 9,   # keto          (G, T)
    'Y': 10, 'y': 10,  # pyrimidine    (C, T)
}

# Total vocabulary size exported so dqn.py / models.py stay in sync automatically
VOCAB_SIZE = 11  # pad(0) + 4 canonical(1-4) + gap(5) + 5 IUPAC(6-10)

nucleotides = ['A', 'T', 'C', 'G', '-']


class Environment:
    def __init__(self, data,
                 nucleotide_size=50, text_size=25,
                 show_nucleotide_name=True):
        self.data = [[nucleotides_map[data[i][j]] for j in range(len(data[i]))] for i in range(len(data))]
        self.row = len(data)
        self.max_len = max([len(data[i]) for i in range(len(data))])
        self.show_nucleotide_name = show_nucleotide_name
        self.nucleotide_size = nucleotide_size
        self.max_window_width = 1800
        self.text_size = text_size

        self.action_number = 2 ** self.row - 1

        self.max_reward = self.row * (self.row - 1) / 2 * config.MATCH_REWARD

        self.aligned = [[] for _ in range(self.row)]
        # Improvement 1: store not_aligned as deques for O(1) front removal
        self.not_aligned = [deque(seq) for seq in copy.deepcopy(self.data)]

    def __action_combination(self):
        res = []
        for i in range(self.row + 1):
            combs = list(combinations(range(self.row), i))

            for j in combs:
                a = np.zeros(self.row)
                for k in j:
                    a[k] = 1
                res.append(a)

        res.pop()

        return res

    def __get_current_state(self):
        state = []
        for i in range(self.row):
            state.extend((self.not_aligned[i][j] if j < len(self.not_aligned[i]) else 5)
                         for j in range(len(self.not_aligned[i]) + 1))

        state.extend([0 for _ in range(self.row * (self.max_len + 1) - len(state))])
        return state

    def __calc_reward(self):
        score = 0
        tail = len(self.aligned[0]) - 1
        for j in range(self.row):
            for k in range(j + 1, self.row):
                if self.aligned[j][tail] == 5 or self.aligned[k][tail] == 5:
                    score += config.GAP_PENALTY
                elif self.aligned[j][tail] == self.aligned[k][tail]:
                    score += config.MATCH_REWARD
                elif self.aligned[j][tail] != self.aligned[k][tail]:
                    score += config.MISMATCH_PENALTY

        return score

    def reset(self):
        self.aligned = [[] for _ in range(self.row)]
        # Improvement 1: reinitialize as deques for O(1) front removal
        self.not_aligned = [deque(seq) for seq in copy.deepcopy(self.data)]
        return self.__get_current_state()

    def step(self, action):
        for bit in range(self.row):
            if 0 == (action >> bit) & 0x1 and 0 == len(self.not_aligned[bit]):
                return -self.max_reward, self.__get_current_state(), 0

        total_len = 0
        for bit in range(self.row):
            if 0 == (action >> bit) & 0x1:
                self.aligned[bit].append(self.not_aligned[bit][0])
                # Improvement 1: O(1) popleft instead of O(n) pop(0)
                self.not_aligned[bit].popleft()
            else:
                self.aligned[bit].append(5)
            total_len += len(self.not_aligned[bit])

        return self.__calc_reward(), self.__get_current_state(), 1 if total_len > 0 else 0

    def calc_score(self):
        score = 0
        for i in range(len(self.aligned[0])):
            for j in range(self.row):
                for k in range(j + 1, self.row):
                    if self.aligned[j][i] == 5 or self.aligned[k][i] == 5:
                        score += config.GAP_PENALTY
                    elif self.aligned[j][i] == self.aligned[k][i]:
                        score += config.MATCH_REWARD
                    elif self.aligned[j][i] != self.aligned[k][i]:
                        score += config.MISMATCH_PENALTY

        return score

    def calc_exact_matched(self):
        score = 0

        for i in range(len(self.aligned[0])):
            n = self.aligned[0][i]
            flag = True
            for j in range(1, self.row):
                if n != self.aligned[j][i]:
                    flag = False
                    break
            if flag:
                score += 1

        return score

    def set_alignment(self, seqs):
        self.aligned = [[nucleotides_map[seqs[i][j]] for j in range(len(seqs[i]))] for i in range(len(seqs))]
        # Keep as empty deques for consistency
        self.not_aligned = [deque() for _ in range(len(self.data))]

    def get_alignment(self):
        alignment = ""
        for i in range(len(self.aligned)):
            alignment += ''.join([nucleotides[self.aligned[i][j] - 1] for j in range(len(self.aligned[i]))]) + '\n'

        return alignment.rstrip()

    def padding(self):
        max_length = 0
        for i in range(len(self.not_aligned)):
            max_length = max(max_length, len(self.not_aligned[i]))

        for i in range(len(self.not_aligned)):
            # deque supports iteration and extend, so this works unchanged
            self.aligned[i].extend(self.not_aligned[i])
            self.aligned[i].extend([5 for _ in range(max_length - len(self.not_aligned[i]))])
            self.not_aligned[i].clear()
