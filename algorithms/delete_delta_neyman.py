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

def delete_delta_neyman_shap(dynamic_game, initial_m, local_m, proc_num=1) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions
    """
    if proc_num < 0:
        raise ValueError('Invalid proc num.')

    n = dynamic_game.n
    dynamic_size = dynamic_game.dynamic_size
    ori_samples = dynamic_game.ori_samples
    sv = np.zeros(n-dynamic_size)
    sv_difference = np.zeros(n-dynamic_size)
    complementary_contributions = [[[] for _ in range(n)] for _ in range(n)]
    complementary_contributions_difference = [[[] for _ in range(n)] for _ in range(n)]
    coef = [comb(n - 1, s) for s in range(n)]
    local_state = np.random.RandomState(None)
    var_dcc = np.zeros((n, n))

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

                del_index = np.array([n-dynamic_size+_ for _ in range(dynamic_size)])

                u_del_left = dynamic_game.eval_utility([x for x in idxs[:j]+[i] if x not in del_index])
                u_del_right = dynamic_game.eval_utility([x for x in idxs[j:] if x not in del_index])

                del_left_len = len([x for x in idxs[:j]+[i] if x not in del_index])
                complementary_contributions[i][del_left_len-1].append(u_del_left-u_del_right)
                complementary_contributions_difference[i][j].append((u_del_left-u_del_right)-(u_left-u_right))
                for l in range(n-1):
                    if l < j:
                        complementary_contributions[idxs[l]][del_left_len-1].append(u_del_left-u_del_right)
                        complementary_contributions_difference[idxs[l]][j].append((u_del_left-u_del_right)-(u_left-u_right))
                    else:
                        complementary_contributions[idxs[l]][n-dynamic_size-del_left_len-1].append(-(u_del_left-u_del_right))
                        complementary_contributions_difference[idxs[l]][n-j-2].append(-((u_del_left-u_del_right)-(u_left-u_right)))
        if count == temp_count:
            break

    for i in range(n):
        for j in range(n):
            var_dcc[i][j] = np.var(complementary_contributions_difference[i][j])
            var_dcc[i][j] = 0 if var_dcc[i][j] == 0 else var_dcc[i][j] * len(
                complementary_contributions_difference[i][j]) / (len(complementary_contributions_difference[i][j]) - 1)
    var_sum = 0
    sigma_dcc_j = np.zeros(n)
    sigma_dcc_n_j = np.zeros(n)
    for j in range(math.ceil(n / 2) - 1, n):
        for i in range(n):
            sigma_dcc_j[j] += var_dcc[i][j] / (j + 1)
            if n - j - 2 < 0:
                sigma_dcc_n_j[j] += 0
            else:
                sigma_dcc_n_j[j] += var_dcc[i][n - j - 2] / (n - j - 1)
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
    func = partial(_delete_delta_neyman_task, dynamic_game)
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

    for i in range(n-dynamic_size):
        for j in range(n):
            sv[i] += 0 if count_cc[i][j] == 0 else (new_complementary_contributions[i][j] / count_cc[i][j])
            sv_difference[i] += 0 if count_difference[i][j] == 0 else(new_complementary_contributions_difference[i][j]/count_difference[i][j])

    sv /= (n-dynamic_size)
    sv_difference /= n

    for i in range(n-dynamic_size):
        sv[i] = dynamic_game.ori_shapley[i]+sv_difference[i]
    return sv

def _delete_delta_neyman_task(dynamic_game,m) -> np.array:
    """
        Compute the difference of complementary contributions and new Shapley value
    """
    n = dynamic_game.n
    dynamic_size = dynamic_game.dynamic_size
    local_state = np.random.RandomState(None)
    complementary_contributions = np.zeros((n, n))
    complementary_contributions_difference = np.zeros((n,n))
    count = np.zeros((n, n))
    count_difference = np.zeros((n, n))
    idxs = np.arange(n)

    for j in range(n):
        for _ in range(m[j]):
            local_state.shuffle(idxs)
            u_left = dynamic_game.eval_utility(idxs[:j+1])
            u_right = dynamic_game.eval_utility(idxs[j+1:])

            del_index = np.array([n-dynamic_size+i for i in range(dynamic_size)])

            u_del_left = dynamic_game.eval_utility([x for x in idxs[:j+1] if x not in del_index])
            u_del_right = dynamic_game.eval_utility([x for x in idxs[j+1:] if x not in del_index])

            temp = np.zeros(n)
            temp[idxs[:j+1]] = 1
            l = len([x for x in idxs[:j+1] if x not in del_index])
            complementary_contributions[:, l-1] += temp * (u_del_left - u_del_right)
            count[:, l-1] += temp
            complementary_contributions_difference[:,j] += temp *[(u_del_left-u_del_right)-(u_left-u_right)]
            count_difference[:,j] += temp

            temp = np.zeros(n)
            temp[idxs[j+1:]] = 1
            l = len([x for x in idxs[j+1:] if x not in del_index])
            if l != 0:
                complementary_contributions[:, l-1] += temp * (u_del_right - u_del_left)
                count[:, l-1] += temp
                if j != n-1:
                    complementary_contributions_difference[:,n-j-2] += temp*[(u_del_right-u_del_left)-(u_right-u_left)]
                    count_difference[:,n-j-2] += temp

    return complementary_contributions, count, complementary_contributions_difference,count_difference





