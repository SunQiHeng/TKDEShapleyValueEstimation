import math
import os
import time
from pathlib import Path
from itertools import chain, combinations
import numpy as np
import pandas as pd
from sklearn import metrics
from typing import Iterator


class Game:
    def __init__(self,
                 game_type,
                 n,
                 w=None,
                 x_train=None,
                 y_train=None,
                 x_test=None,
                 y_test=None,
                 model=None):
        self.game_type = game_type
        self.n = n
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model

        if w is None:
            if game_type == 'voting':
                self.w = [
                    45, 41, 27, 26, 26, 25, 21, 17, 17, 14, 13, 13, 12, 12, 12,
                    11, 10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 7, 7, 7, 7, 6, 6, 6,
                    6, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 45, 41, 27, 26, 26, 25, 21, 17, 17, 14, 13, 13, 12, 12, 12,
                    11, 10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 7, 7, 7, 7, 6, 6, 6,
                    6, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3
                ]
            elif game_type == 'airport':
                self.w = [
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6,
                    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9
                ]
        else:
            self.w = load_npy(w)
            if game_type == 'voting':
                self.hw = np.sum(self.w) / 2
                w_sorted = sorted(self.w, reverse=True)
                s = 0
                self.i = 0
                while s < self.hw and self.i < n:
                    s += w_sorted[self.i]
                    self.i += 1

        if game_type != 'tree' and game_type != 'model':
            self.w = np.array(self.w)

    def eval_utility(self, x):
        """Evaluate the coalition utility.
        """
        if len(x) == 0:
            return 0
        if self.game_type == 'voting':
            coalition_sum = np.sum(self.w[x])
            all_sum = np.sum(self.w[:self.n])
            return 1 if coalition_sum > all_sum/2 else 0
        elif self.game_type == 'airport':
            colation_max = np.max(self.w[x])
            return colation_max
        elif self.game_type == 'tree':
            r = self.n
            x = np.sort(x)
            m = r - (x[-1] - x[0])
            for i in range(1, len(x)):
                t = x[i] - x[i - 1]
                if t > m:
                    m = t
            return r - m
        elif self.game_type == 'model':
            temp_x, temp_y = self.x_train[x], self.y_train[x]
            single_pred_label = (True if len(np.unique(temp_y)) == 1 else False)
            if single_pred_label:
                y_pred = [temp_y[0]] * len(self.y_test)
            else:
                try:
                    self.model.fit(temp_x, temp_y)
                    y_pred = self.model.predict(self.x_test)
                except:
                    return None
            return metrics.accuracy_score(self.y_test, y_pred, normalize=True)

class DynamicGame(Game):
    def __init__(self,
                 game_type,
                 n,
                 dynamic_size,
                 ori_shapley,
                 ori_samples,
                 w=None,
                 x_train=None,
                 y_train=None,
                 x_test=None,
                 y_test=None,
                 model=None
                 ):
        super().__init__(
                 game_type,
                 n,
                 w=w,
                 x_train=x_train,
                 y_train=y_train,
                 x_test=x_test,
                 y_test=y_test,
                 model=model,
                 )
        self.dynamic_size = dynamic_size
        self.ori_shapley = ori_shapley
        self.ori_samples = ori_samples

    def eval_utility(self, x):
        return super().eval_utility(x)