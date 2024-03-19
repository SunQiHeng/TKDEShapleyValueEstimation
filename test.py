import sys,os
path = os.path.dirname("/home/sunqiheng/Code/PyCode/ShapleyValueEstimation/sqhsrc")
sys.path.append(path)

from sqhsrc.algorithms.add_delta import add_delta_shap
from sqhsrc.algorithms.cc import cc_shap
from sqhsrc.algorithms.mcn import mcn
from sqhsrc.algorithms.cc_neyman import ccshap_neyman
from sqhsrc.utils.game_class import Game,DynamicGame
from sqhsrc.algorithms.add_delta_neyman import add_delta_neyman_shap
import matplotlib.pyplot as plt
from sqhsrc.utils.options import args_parser
from sqhsrc.utils.tools import AverageErrorRatio,preprocess_adult,preprocess_wdbc,preprocess_bank_marketing
import math
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd

args = args_parser()
args.game = 'model'
X_train, Y_train, X_test, Y_test = preprocess_bank_marketing()
args.players = 600
classifier = SVC()

# 使用训练数据拟合分类器
classifier.fit(X_train, Y_train)

# 在测试集上进行预测
Y_pred = classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")




