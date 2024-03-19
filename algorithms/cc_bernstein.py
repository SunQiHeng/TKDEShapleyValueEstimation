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

def ccshap_bernstein(game,
                     local_m,
                     initial_m,
                     r,
                     thelta,
                     flag=False) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions based on Bernstein
    """
    n = game.n

    def select_coalitions_on(rankings):
        n = len(rankings)
        idxs_set = set([i for i in range(n)])
        res_idx = []
        res_s = math.inf
        max_score = 0
        for s in range(math.ceil(n / 2), n - 1):
            perm = [
                rankings[i][n - s - 1] - rankings[i][s - 1] for i in range(n)
            ]
            tidxs = np.argsort(perm)
            rank = perm[tidxs[s - 1]]
            local_state.shuffle(tidxs)
            idxs = []
            for i in tidxs:
                if perm[i] > rank:
                    idxs.append(tidxs[i])
            i = 0
            while len(idxs) < n - s and i < n:
                if perm[tidxs[i]] == rank:
                    idxs.append(tidxs[i])
                i += 1
            idxs += list(idxs_set - set(idxs))
            score = 0
            for i in range(n - s):
                score += rankings[idxs[i]][n - s - 1]
            for i in range(n - s, n):
                score += rankings[idxs[i]][s - 1]
            if score > max_score:
                max_score = score
                res_idx, res_s = idxs, s
        return max_score, res_idx, res_s

    # use original algorithm and a little random
    def select_coalitions_rd(rankings):
        local_state = np.random.RandomState(None)
        n = len(rankings)

        rankings_matrix = rankings

        idx = np.arange(n)
        max_score = 0
        for s in range(math.ceil(n / 2), n):
            local_state.shuffle(idx)

            l = []
            fl = []
            for i in range(s):
                temp = rankings_matrix[idx[i]][s - 1] - rankings_matrix[
                    idx[i]][n - s - 1]
                l.append(temp)
                fl.append(-temp)

            r = []
            fr = []
            for i in range(s, n):
                temp = rankings_matrix[idx[i]][n - s -
                                               1] - rankings_matrix[idx[i]][s -
                                                                            1]
                r.append(temp)
                fr.append(-temp)

            sli = np.argsort(fl)
            slr = np.argsort(fr)

            score = 0
            p = 0
            while p < s and p < n - s and score + l[sli[p]] + r[slr[p]] < score:
                score += l[sli[p]] + r[slr[p]]
                idx[sli[p]], idx[s + slr[p]] = idx[s + slr[p]], idx[sli[p]]
                p += 1

            score = 0
            for i in range(s):
                score += rankings_matrix[idx[i]][s - 1]
            for i in range(s, n):
                score += rankings_matrix[idx[i]][n - s - 1]

            if score > max_score:
                max_score = score
                res_idx, res_s = idx, s

        return max_score, res_idx, res_s

    local_state = np.random.RandomState(None)
    utility = [[[] for _ in range(n)] for _ in range(n)]
    var = np.zeros((n, n))
    coef = [comb(n - 1, s) for s in range(n)]
    # thelta = np.log(10 / (1 + thelta))

    if flag:
        print("prob: ", thelta)
    else:
        print("prob: ", 1 - 3 * np.power(math.e, -thelta))

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

    for i in range(n):
        for j in range(n):
            if flag:
                var[i][j] = np.var(utility[i][j]) * len(
                    utility[i][j]) / (len(utility[i][j]) - 1)
                var[i][j] = (np.sqrt(2 * var[i][j] * math.log(2 / thelta) /
                                     len(utility[i][j])) +
                             7 * math.log(2 / thelta) /
                             (3 * (len(utility[i][j]) - 1)))
            else:
                var[i][j] = np.var(utility[i][j])
                var[i][j] = (
                    np.sqrt(2 * var[i][j] * thelta / len(utility[i][j])) +
                    3 * r * thelta / len(utility[i][j]))

    count = sum([len(utility[0][j]) for j in range(n)])
    local_m -= count
    print(local_m)
    for _ in trange(local_m):
        min_score, idxs, j = select_coalitions_rd(var)

        u_1 = game.eval_utility(idxs[:j])
        u_2 = game.eval_utility(idxs[j:])
        for l in range(n):
            if l < j:
                utility[idxs[l]][j - 1].append(u_1 - u_2)
                if flag:
                    var[idxs[l]][j -
                                 1] = np.var(utility[idxs[l]][j - 1]) * len(
                                     utility[idxs[l]][j - 1]) / (
                                         len(utility[idxs[l]][j - 1]) - 1)
                    var[idxs[l]][
                        j -
                        1] = 1 * (np.sqrt(2 * var[idxs[l]][j - 1] * math.log(
                            2 / thelta) / len(utility[idxs[l]][j - 1])) +
                                  7 * math.log(2 / thelta) /
                                  (3 * (len(utility[idxs[l]][j - 1]) - 1)))
                else:
                    var[idxs[l]][j - 1] = np.var(utility[idxs[l]][j - 1])
                    var[idxs[l]][j - 1] = 1 * (
                        np.sqrt(2 * var[idxs[l]][j - 1] * thelta /
                                len(utility[idxs[l]][j - 1])) +
                        3 * 1 * thelta / len(utility[idxs[l]][j - 1]))
            else:
                utility[idxs[l]][n - j - 1].append(u_2 - u_1)
                if flag:
                    var[idxs[l]][n - j - 1] = np.var(
                        utility[idxs[l]][n - j - 1]) * len(
                            utility[idxs[l]][n - j - 1]) / (
                                len(utility[idxs[l]][n - j - 1]) - 1)
                    var[idxs[l]][n - j - 1] = 1 * (
                        np.sqrt(2 * var[idxs[l]][n - j - 1] * math.log(
                            2 / thelta) / len(utility[idxs[l]][n - j - 1])) +
                        7 * math.log(2 / thelta) /
                        (3 * (len(utility[idxs[l]][n - j - 1]) - 1)))
                else:
                    var[idxs[l]][n - j - 1] = np.var(utility[idxs[l]][n - j -
                                                                      1])
                    var[idxs[l]][n - j - 1] = 1 * (
                        np.sqrt(2 * var[idxs[l]][n - j - 1] * thelta /
                                len(utility[idxs[l]][n - j - 1])) +
                        3 * 1 * thelta / len(utility[idxs[l]][n - j - 1]))

    sv = np.zeros(n)
    for i in range(n):
        for j in range(n):
            sv[i] += np.mean(utility[i][j])
        sv[i] /= n
    return sv