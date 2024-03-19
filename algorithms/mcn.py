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
                    
def mcn(game, m, proc_num=1) -> np.ndarray:
    """Compute the Shapley value by sampling marginal contributions by optimum allocation
    """
    n = game.n
    var = np.zeros((n, n))
    utility = [[[] for i in range(n)] for i in range(n)]
    count = np.zeros((n, n))
    initial_m = max(2, int(m / (2 * n * n)))
    args = split_permutation(n, proc_num)
    pool = Pool()
    func = partial(_mcn_initial, game, initial_m)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    for r in ret:
        for i in range(n):
            for j in range(n):
                utility[i][j].extend(r[0][i][j])
        count += r[1]

    for i in range(n):
        for j in range(n):
            var[i][j] = np.var(utility[i][j]) * count[i][j] / (count[i][j] - 1)

    mst = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mst[i][j] = max(0, m * var[i][j] / np.sum(var) - count[i][j])

    args = [[] for i in range(proc_num)]
    for _ in mst:
        temp_args = split_num(_, proc_num).tolist()
        for i in range(len(temp_args)):
            args[i].append(temp_args[i])

    pool = Pool()
    func = partial(_mcn_task, game)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    for r in ret:
        for i in range(n):
            for j in range(n):
                utility[i][j].extend(r[i][j])

    sv = np.zeros(n)
    for i in range(n):
        for j in range(n):
            sv[i] += np.mean(utility[i][j])
        sv[i] /= n
    return sv


def _mcn_initial(game, initial_m, i_list):
    n = game.n
    local_state = np.random.RandomState(None)
    utility = [[[] for i in range(n)] for i in range(n)]
    count = np.zeros((n, n))
    for h in trange(len(i_list)):
        i = i_list[h]
        idxs = []
        for _ in range(n):
            if _ is not i:
                idxs.append(_)
        for j in range(n):
            for _ in range(initial_m):
                local_state.shuffle(idxs)
                u_1 = game.eval_utility(idxs[:j])
                u_2 = game.eval_utility(idxs[:j] + [i])
                utility[i][j].append(u_2 - u_1)
                count[i][j] += 1
    return utility, count


def _mcn_task(game, mst):
    n = game.n
    local_state = np.random.RandomState(None)
    utility = [[[] for i in range(n)] for i in range(n)]
    for i in trange(n):
        idxs = []
        for _ in range(n):
            if _ is not i:
                idxs.append(_)
        for j in range(n):
            for _ in range(mst[i][j]):
                local_state.shuffle(idxs)
                u_1 = game.eval_utility(idxs[:j])
                u_2 = game.eval_utility(idxs[:j] + [i])
                utility[i][j].append(u_2 - u_1)
    return utility