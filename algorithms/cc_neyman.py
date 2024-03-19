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

def ccshap_neyman(game, initial_m, local_m, proc_num=1) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions based on Neyman
    """
    n = game.n
    sv = np.zeros(n)
    utility = [[[] for _ in range(n)] for _ in range(n)]
    var = np.zeros((n, n))
    local_state = np.random.RandomState(None)
    coef = [comb(n - 1, s) for s in range(n)]

    # initialize
    count = 0
    while True:
        temp_count = count
        for i in trange(n):
            idxs = [_ for _ in range(i)] + [_ for _ in range(i + 1, n)]
            for j in range(n):
                if len(utility[i][j]) >= initial_m or len(
                        utility[i][j]) >= coef[j]:
                    continue
                local_state.shuffle(idxs)
                count += 1
                u_1 = game.eval_utility(idxs[:j] + [i])
                u_2 = game.eval_utility(idxs[j:])
                utility[i][j].append(u_1 - u_2)
                for l in range(n - 1):
                    if l < j:
                        utility[idxs[l]][j].append(u_1 - u_2)
                    else:
                        utility[idxs[l]][n - j - 2].append(u_2 - u_1)

        if count == temp_count:
            break

    # compute allocation
    for i in range(n):
        for j in range(n):
            var[i][j] = np.var(utility[i][j])
            var[i][j] = 0 if var[i][j] == 0 else var[i][j] * len(
                utility[i][j]) / (len(utility[i][j]) - 1)

    var_sum = 0
    sigma_j = np.zeros(n)
    sigma_n_j = np.zeros(n)
    for j in range(math.ceil(n / 2) - 1, n):
        for i in range(n):
            sigma_j[j] += var[i][j] / (j + 1)
            if n - j - 2 < 0:
                sigma_n_j[j] += 0
            else:
                sigma_n_j[j] += var[i][n - j - 2] / (n - j - 1)
        var_sum += np.sqrt(sigma_j[j] + sigma_n_j[j])

    local_m -= count
    m = np.zeros(n)
    for j in range(math.ceil(n / 2) - 1, n):
        m[j] = max(0, math.ceil(local_m * np.sqrt(sigma_j[j] + sigma_n_j[j]) / var_sum))

    args = split_num(m, proc_num)
    pool = Pool()
    func = partial(_ccshap_neyman_task, game)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    new_utility = np.zeros((n, n))
    count = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            new_utility[i][j] = np.sum(utility[i][j])
            count[i][j] = len(utility[i][j])

    for r in ret:
        new_utility += r[0]
        count += r[1]

    for i in range(n):
        for j in range(n):
            sv[i] += 0 if count[i][j] == 0 else new_utility[i][j] / count[i][j]
        sv[i] /= n
    return sv


def _ccshap_neyman_task(game, m):
    n = game.n
    utility = np.zeros((n, n))
    count = np.zeros((n, n))
    idxs = np.arange(n)
    local_state = np.random.RandomState(None)
    for j in trange(n):
        for _ in range(m[j]):
            local_state.shuffle(idxs)
            u_1 = game.eval_utility(idxs[:j + 1])
            u_2 = game.eval_utility(idxs[j + 1:])

            temp = np.zeros(n)
            temp[idxs[:j + 1]] = 1
            utility[:, j] += temp * (u_1 - u_2)
            count[:, j] += temp

            temp = np.zeros(n)
            temp[idxs[j + 1:]] = 1
            utility[:, n - j - 2] += temp * (u_2 - u_1)
            count[:, n - j - 2] += temp

            # for l in range(n):
            #     if l < j + 1:
            #         utility[idxs[l]][j] += (u_1 - u_2)
            #         count[idxs[l]][j] += 1
            #     else:
            #         utility[idxs[l]][n - j - 2] += (u_2 - u_1)
            #         count[idxs[l]][n - j - 2] += 1

    return utility, count


def ccshap_neyman_n(game, initial_m, local_m) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions based on Neyman
    """
    n = game.n
    sv = np.zeros(n)
    utility = [[[] for _ in range(n)] for _ in range(n)]
    var = np.zeros((n, n))
    local_state = np.random.RandomState(None)
    coef = [comb(n - 1, s) for s in range(n)]

    # initialize
    count = 0
    while True:
        temp_count = count
        for i in trange(n):
            idxs = [_ for _ in range(i)] + [_ for _ in range(i + 1, n)]
            for j in range(n):
                if len(utility[i][j]) >= initial_m or len(
                        utility[i][j]) >= coef[j]:
                    continue
                local_state.shuffle(idxs)
                count += 1
                u_1 = game.eval_utility(idxs[:j] + [i])
                u_2 = game.eval_utility(idxs[j:])
                utility[i][j].append(u_1 - u_2)
                for l in range(n - 1):
                    if l < j:
                        utility[idxs[l]][j].append(u_1 - u_2)
                    else:
                        utility[idxs[l]][n - j - 2].append(u_2 - u_1)

        if count == temp_count:
            break

    # compute allocation
    for i in range(n):
        for j in range(n):
            var[i][j] = np.var(utility[i][j])
            var[i][j] = 0 if var[i][j] == 0 else var[i][j] * len(
                utility[i][j]) / (len(utility[i][j]) - 1)

    var_sum = 0
    sigma_j = np.zeros(n)
    sigma_n_j = np.zeros(n)
    for j in range(math.ceil(n / 2) - 1, n):
        for i in range(n):
            sigma_j[j] += var[i][j] / (j + 1)
            if n - j - 2 < 0:
                sigma_n_j[j] += 0
            else:
                sigma_n_j[j] += var[i][n - j - 2] / (n - j - 1)
        var_sum += np.sqrt(sigma_j[j] + sigma_n_j[j])

    local_m -= count
    print(local_m)
    m = np.zeros(n)
    for j in range(math.ceil(n / 2) - 1, n):
        m[j] = max(
            0,
            math.ceil(local_m * np.sqrt(sigma_j[j] + sigma_n_j[j]) / var_sum))

    new_utility = np.zeros((n, n))
    count = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            new_utility[i][j] = np.sum(utility[i][j])
            count[i][j] = len(utility[i][j])

    idxs = np.arange(n)
    for j in trange(n):
        for _ in range(int(m[j])):
            local_state.shuffle(idxs)
            u_1 = game.eval_utility(idxs[:j + 1])
            u_2 = game.eval_utility(idxs[j + 1:])

            temp = np.zeros(n)
            temp[idxs[:j + 1]] = 1
            new_utility[:, j] += temp * (u_1 - u_2)
            count[:, j] += temp

            temp = np.zeros(n)
            temp[idxs[j + 1:]] = 1
            new_utility[:, n - j - 2] += temp * (u_2 - u_1)
            count[:, n - j - 2] += temp

    for i in range(n):
        for j in range(n):
            sv[i] += 0 if count[i][j] == 0 else new_utility[i][j] / count[i][j]
        sv[i] /= n

    return sv