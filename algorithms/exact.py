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


def exact_shap_n(game, log_step):
    """Compute the exact Shapley value
    """
    n = game.n

    def add(l):
        for i in range(n - 1, -1, -1):
            if l[i] == 0:
                l[i] = 1
                break
            else:
                l[i] = 0
        return l

    ext_sv = np.zeros(n)
    coef = np.zeros(n)
    fact = np.math.factorial
    coalition = np.arange(n)
    for s in range(n):
        coef[s] = fact(s) * fact(n - s - 1) / fact(n)
    l = np.zeros(n)

    j = 0
    add(l)
    while np.sum(l) != 0:
        if j % log_step == 0:
            print(j)
        j += 1
        idx = []
        for i in range(n):
            if l[i] == 1:
                idx.append(i)
        u = game.eval_utility(idx)
        for i in idx:
            ext_sv[i] += coef[len(idx) - 1] * u
        for i in set(coalition) - set(idx):
            ext_sv[i] -= coef[len(idx)] * u
        add(l)

    return ext_sv