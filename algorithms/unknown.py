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


def add_delta_change_neyman_shap(dynamic_game, initial_m, local_m, proc_num=1) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions
    """
    if proc_num < 0:
        raise ValueError('Invalid proc num.')

    n = dynamic_game.n
    sv = np.zeros(n)
    sv_difference = np.zeros(n)
    complementary_contributions = [[[] for _ in range(n)] for _ in range(n)]
    complementary_contributions_difference = [[[] for _ in range(n)] for _ in range(n)]
    coef = [comb(n - 1, s) for s in range(n)]
    local_state = np.random.RandomState(None)
    var_dcc = np.zeros((n, n))
    var_cc = np.zeros((n, n))
    # initialize 
    count = 0
    while True:
        temp_count = count
        for i in range(n):
            idxs = [_ for _ in range(i)] + [_ for _ in range(i + 1, n)]
            for j in range(n):
                if len(complementary_contributions_difference[i][j]) >= initial_m or len(complementary_contributions_difference[i][j]) >= coef[j]:
                    continue
                count += 1
                local_state.shuffle(idxs)
                u_left = dynamic_game.eval_utility(idxs[:j]+[i])
                u_right = dynamic_game.eval_utility(idxs[j:])
                add_index = np.array([dynamic_game.n-dynamic_game.dynamic_size+_ for _ in range(dynamic_game.dynamic_size)])
                u_original_left = dynamic_game.eval_utility([x for x in idxs[:j]+[i] if x not in add_index])
                u_original_right = dynamic_game.eval_utility([x for x in idxs[j:] if x not in add_index])
                complementary_contributions[i][j].append(u_left-u_right)
                complementary_contributions_difference[i][j].append((u_left-u_right)-(u_original_left-u_original_right))
                for l in range(n-1):
                    if l <j:
                        complementary_contributions[idxs[l]][j].append(u_left-u_right)
                        complementary_contributions_difference[idxs[l]][j].append((u_left-u_right)-(u_original_left-u_original_right))
                    else:
                        complementary_contributions[idxs[l]][n-j-2].append(-(u_left-u_right))
                        complementary_contributions_difference[idxs[l]][n-j-2].append(-((u_left-u_right)-(u_original_left-u_original_right)))
        if count == temp_count:
            break
  
    for i in range(n):
        for j in range(n):
            var_dcc[i][j] = np.var(complementary_contributions_difference[i][j])
            var_dcc[i][j] = 0 if var_dcc[i][j] == 0 else var_dcc[i][j] * len(
                complementary_contributions_difference[i][j]) / (len(complementary_contributions_difference[i][j]) - 1)
            var_cc[i][j] = np.var(complementary_contributions[i][j])
            var_cc[i][j] = 0 if var_cc[i][j] == 0 else var_cc[i][j] * len(complementary_contributions[i][j])/(len(complementary_contributions[i][j])-1)
    var_sum = 0
    sigma_dcc_j = np.zeros(n)
    sigma_dcc_n_j = np.zeros(n)
    sigma_cc_j = np.zeros(n)
    sigma_cc_n_j = np.zeros(n)
    for j in range(math.ceil(n / 2) - 1, n):
        for i in range(n):
            if i < n-dynamic_game.dynamic_size:
                sigma_dcc_j[j] += var_dcc[i][j] / (j + 1)
                if n - j - 2 < 0:
                    sigma_dcc_n_j[j] += 0
                else:
                    sigma_dcc_n_j[j] += var_dcc[i][n - j - 2] / (n - j - 1)
            else:
                sigma_cc_j[j] += var_cc[i][j] / (j + 1)
                if n - j - 2 < 0:
                    sigma_cc_n_j[j] += 0
                else:
                    sigma_cc_n_j[j] += var_cc[i][n - j - 2]/(n - j - 1)
        var_sum += np.sqrt(sigma_dcc_j[j] + sigma_dcc_n_j[j])

    local_m -= 2*count
    m = np.zeros(n)
    if var_sum != 0:
        for j in range(math.ceil(n / 2) - 1, n):
            m[j] = max(0, math.ceil(local_m/2* np.sqrt(sigma_dcc_j[j] + sigma_dcc_n_j[j]) / var_sum))
    else:
        for j in range(math.ceil(n / 2) - 1, n):
            m[j] = max(0, math.ceil(local_m/2/ (n-math.ceil(n/2)+1)))

    args = split_num(m, proc_num)
    pool = Pool()
    func = partial(_add_delta_change_neyman_task, dynamic_game)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    new_complementary_contributions = np.zeros((n,n))
    count_cc = np.zeros((n,n))
    new_complementary_contributions_difference = np.zeros((n,n))
    count_difference = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            new_complementary_contributions[i][j] = np.sum(complementary_contributions[i][j])
            count_cc[i][j] = len(complementary_contributions[i][j])
            new_complementary_contributions_difference[i][j] = np.sum(complementary_contributions_difference[i][j])
            count_difference[i][j] = len(complementary_contributions_difference[i][j])

    for r in ret:
        new_complementary_contributions += r[0]
        count_cc += r[1]
        new_complementary_contributions_difference += r[2]
        count_difference += r[3]

    for i in range(n):
        for j in range(n):
            sv[i] += 0 if count_cc[i][j] == 0 else (new_complementary_contributions[i][j] / count_cc[i][j])
            sv_difference[i] += 0 if count_difference[i][j] == 0 else (new_complementary_contributions_difference[i][j]/count_difference[i][j])
    sv /= n
    sv_difference /= n
    for i in range(dynamic_game.n-dynamic_game.dynamic_size):
        sv[i] = dynamic_game.ori_shapley[i]+sv_difference[i]
    return sv

def _add_delta_change_neyman_task(dynamic_game,m) -> np.array:
    """
        Compute the difference of complementary contributions and new Shapley value
    """
    n = dynamic_game.n
    local_state = np.random.RandomState(None)
    complementary_contributions = np.zeros((n, n))
    complementary_contributions_difference = np.zeros((n ,n))
    count = np.zeros((n, n))
    count_difference = np.zeros((n, n))
    idxs = np.arange(n)

    for j in trange(n):
        for _ in range(m[j]):
            local_state.shuffle(idxs)
            u_left = dynamic_game.eval_utility(idxs[:j+1])
            u_right = dynamic_game.eval_utility(idxs[j+1:])

            add_index = np.array([dynamic_game.n-dynamic_game.dynamic_size+_ for _ in range(dynamic_game.dynamic_size)])

            u_original_left = dynamic_game.eval_utility([x for x in idxs[:j+1] if x not in add_index])
            u_original_right = dynamic_game.eval_utility([x for x in idxs[j+1:] if x not in add_index])

            temp = np.zeros(n)
            temp[idxs[:j+1]] = 1
            complementary_contributions[:,j] += temp * (u_left - u_right)
            count[:, j] += temp
            complementary_contributions_difference[:,j] += temp *[(u_left-u_right)-(u_original_left-u_original_right)]
            count_difference[:,j] += temp

            if j != n-1:
                temp = np.zeros(n)
                temp[idxs[j+1:]] = 1
                complementary_contributions[:, n-j-2] += temp * (u_right - u_left)
                count[:, n-j-2] += temp
                complementary_contributions_difference[:,n-j-2] += temp*[(u_right-u_left)-(u_original_right-u_original_left)]
                count_difference[:,n-j-2] += temp

    return complementary_contributions, count, complementary_contributions_difference,count_difference






