import sys,os
path = os.path.dirname("/home/sunqiheng/Code/PyCode/ShapleyValueEstimation/sqhsrc")
sys.path.append(path)

import datetime
import math
import random
from functools import partial
from multiprocessing import Pool
import numpy as np
from scipy.special import comb
from tqdm import trange
import copy

from sqhsrc.utils.tools import (split_permutation_num, split_permutation, split_num,
                    power_set)
                    

def mch_n(game, m) -> np.ndarray:
    """Compute the Shapley value by sampling marginal contributions based on hoeffding
    """
    n = game.n
    sv = np.zeros(n)
    mk = np.zeros(n)
    mk_list = []
    for i in range(n):
        mk_list.append(np.power(i + 1, 2 / 3))
    s = sum(mk_list)
    for i in range(n):
        mk[i] = math.ceil(m * mk_list[i] / s)

    print("mk:", sum(mk))

    local_state = np.random.RandomState(None)
    utility = np.zeros((n, n))
    for i in trange(n):
        idxs = [_ for _ in range(i)] + [_ for _ in range(i + 1, n)]
        for j in range(n):
            for _ in range(int(mk[j])):
                local_state.shuffle(idxs)
                u_1 = game.eval_utility(idxs[:j])
                u_2 = game.eval_utility(idxs[:j] + [i])
                utility[i][j] += (u_2 - u_1)

    for i in range(n):
        for j in range(n):
            sv[i] += utility[i][j] / mk[j]
        sv[i] /= n

    return sv