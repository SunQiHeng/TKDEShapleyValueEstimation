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


def mc_shap(game, m, proc_num=1) -> np.ndarray:
    """Compute the Monte Carlo Shapley value by sampling permutations
    """
    if proc_num < 0:
        raise ValueError('Invalid proc num.')
    args = split_permutation_num(m, proc_num)
    pool = Pool()
    func = partial(_mc_shap_task, game)
    ret = pool.map(func, args)
    pool.close()
    pool.join()
    ret_arr = np.asarray(ret)
    return np.sum(ret_arr, axis=0) / m


def _mc_shap_task(game, local_m) -> np.ndarray:
    """Compute Shapley value by sampling local_m permutations
    """
    n = game.n
    local_state = np.random.RandomState(None)
    sv = np.zeros(n)
    idxs = np.arange(n)

    for _ in trange(local_m):
        local_state.shuffle(idxs)
        old_u = game.null
        for j in range(1, n + 1):
            temp_u = game.eval_utility(idxs[:j])
            contribution = temp_u - old_u
            sv[idxs[j - 1]] += contribution
            old_u = temp_u

    return sv


def mc_shap_task(game, local_m) -> np.ndarray:
    """Compute Shapley value by sampling local_m permutations
    """
    local_state = np.random.RandomState(None)
    n = game.n
    sv = np.zeros(n)
    idxs = np.arange(n)

    for _ in trange(local_m):
        local_state.shuffle(idxs)
        old_u = game.null  # game.null()[0]
        for j in range(1, n + 1):
            temp_u = game.eval_utility(idxs[:j])
            contribution = temp_u - old_u
            sv[idxs[j - 1]] += contribution
            old_u = temp_u

    return sv / local_m