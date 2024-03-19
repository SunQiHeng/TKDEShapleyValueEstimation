import sys,os
path = os.path.dirname("/home/sunqiheng/Code/PyCode/ShapleyValueEstimation/sqhsrc")
sys.path.append(path)

from sqhsrc.algorithms.add_delta import add_delta_shap
from sqhsrc.algorithms.cc import cc_shap
from sqhsrc.algorithms.mcn import mcn
from sqhsrc.algorithms.cc_neyman import ccshap_neyman
from sqhsrc.utils.game_class import Game,DynamicGame
from sqhsrc.algorithms.add_delta_neyman import add_delta_neyman_shap
from sqhsrc.algorithms.add_delta_change_neyman import add_delta_change_neyman_shap
import matplotlib.pyplot as plt
from sqhsrc.utils.options import args_parser
from sqhsrc.utils.tools import AverageErrorRatio,preprocess_adult,preprocess_bank_marketing,write_to_csv
import math
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import csv
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd


X_train,Y_train,X_test,Y_test = preprocess_bank_marketing()
args = args_parser()
args.game = 'model'

args.ori_players = 490
args.new_players = 500
args.all_players = 500
args.samples = 50*args.new_players
model = SVC()

game = Game(game_type=args.game,n=args.new_players,
    w=None,x_train= X_train, x_test= X_test, y_train = Y_train, y_test = Y_test, model=model)
base_sv = cc_shap(game,25000*args.new_players,proc_num=4)

with open('./sv_data/svm+bank+500.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for item in base_sv:
        writer.writerow([item])









