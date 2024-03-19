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

def delete_delta_shap(dynamic_game, m, proc_num=1) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions
    """
    if proc_num < 0:
        raise ValueError('Invalid proc num.')

    n = dynamic_game.n
    dynamic_size = dynamic_game.dynamic_size
    ori_samples = dynamic_game.ori_samples

    args = split_permutation_num(m, proc_num)
    pool = Pool()
    func = partial(_delete_delta_task, dynamic_game)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    sv = np.zeros(n-dynamic_size)
    sv_difference = np.zeros(n-dynamic_size)
    complementary_contributions = np.zeros((n + 1, n))
    count = np.zeros((n + 1, n))
    complementary_contributions_difference = np.zeros((n + 1, n))
    contributions = np.zeros((n + 1, n))
    count_difference = np.zeros((n + 1, n))
    for r in ret:
        complementary_contributions += r[0]
        count += r[1]
        complementary_contributions_difference += r[2]
        count_difference += r[3]

    for i in range(n + 1):
        for j in range(n-dynamic_size):
            sv[j] += 0 if count[i][j] == 0 else (complementary_contributions[i][j] / count[i][j])
            sv_difference[j] += 0 if count_difference[i][j] == 0 else(complementary_contributions_difference[i][j]/count_difference[i][j])

    sv /= (n-dynamic_size)
    sv_difference /= n

    for i in range(n-dynamic_size):
        #sv[i] = (ori_samples/(m+ori_samples))*(dynamic_game.ori_shapley[i]+sv_difference[i])+(m/(m+ori_samples))*sv[i]
        sv[i] = dynamic_game.ori_shapley[i]+sv_difference[i]
    return sv

def _delete_delta_task(dynamic_game,local_m) -> np.array:
    """
        Compute the difference of complementary contributions and new Shapley value
    """
    n = dynamic_game.n
    dynamic_size = dynamic_game.dynamic_size
    local_state = np.random.RandomState(None)
    complementary_contributions = np.zeros((n + 1, n))
    complementary_contributions_difference = np.zeros((n+1,n))
    count = np.zeros((n + 1, n))
    count_difference = np.zeros((n + 1, n))
    idxs = np.arange(n)

    for _ in trange(int(local_m/2)):
        local_state.shuffle(idxs)
        j = random.randint(1, n)
        u_left = dynamic_game.eval_utility(idxs[:j])
        u_right = dynamic_game.eval_utility(idxs[j:])

        del_index = np.array([n-dynamic_size+i for i in range(dynamic_size)])

        u_del_left = dynamic_game.eval_utility([x for x in idxs[:j] if x not in del_index])
        u_del_right = dynamic_game.eval_utility([x for x in idxs[j:] if x not in del_index])

        temp = np.zeros(n)
        temp[idxs[:j]] = 1
        l = len([x for x in idxs[:j] if x not in del_index])
        complementary_contributions[l, :] += temp * (u_del_left - u_del_right)
        count[l, :] += temp
        complementary_contributions_difference[j,:] += temp *[(u_del_left-u_del_right)-(u_left-u_right)]
        count_difference[j,:] += temp

        temp = np.zeros(n)
        temp[idxs[j:]] = 1
        l = len([x for x in idxs[j:] if x not in del_index])
        complementary_contributions[l, :] += temp * (u_del_right - u_del_left)
        count[l, :] += temp
        complementary_contributions_difference[n-j,:] += temp*[(u_del_right-u_del_left)-(u_right-u_left)]
        count_difference[n-j,:] += temp

    return complementary_contributions, count, complementary_contributions_difference,count_difference





