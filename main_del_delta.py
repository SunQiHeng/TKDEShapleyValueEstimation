import sys,os
path = os.path.dirname("/home/sunqiheng/Code/PyCode/ShapleyValueEstimation/sqhsrc")
sys.path.append(path)

from sqhsrc.algorithms.add_delta import add_delta_shap
from sqhsrc.algorithms.cc import cc_shap
from sqhsrc.algorithms.mcn import mcn
from sqhsrc.algorithms.cc_neyman import ccshap_neyman
from sqhsrc.utils.game_class import Game,DynamicGame
from sqhsrc.algorithms.mc_delta import DynaShap,DeltaShap
from sqhsrc.algorithms.add_delta_neyman import add_delta_neyman_shap
#from sqhsrc.algorithms.add_delta_change_neyman import add_delta_change_neyman_shap
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

def read_csv_file(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = []
        for row in reader:
            data.append(row)
    return np.array(data).astype(float)

ori_sv = read_csv_file('./sv_data/svm+bank+500.csv')
base_sv = read_csv_file('./sv_data/svm+bank+490.csv')

results = []
for i in range(0,6):
    args.samples = (10+2*i)*args.new_players*args.new_players
    McDelta = ori_sv
    pre_sv = ori_sv
    for j in range(10):
        dynashap = DeltaShap(X_train[:500-j],Y_train[:500-j],X_test,Y_test, model, pre_sv)
        temp_sv = dynashap.del_single_point(500-j-1, m=int(args.samples/args.new_players/2/10))
        pre_sv = temp_sv
        McDelta = temp_sv
    temp = []
    temp.append(AverageErrorRatio(base_sv,McDelta))
    results.append(temp)

write_to_csv(results, './exp_results/bank_svm_del_McDelta.csv')










