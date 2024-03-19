import sys,os
path = os.path.dirname("/home/sunqiheng/Code/PyCode/ShapleyValueEstimation/sqhsrc")
sys.path.append(path)

from sqhsrc.algorithms.delete_delta import delete_delta_shap
from sqhsrc.algorithms.delete_delta_neyman import delete_delta_neyman_shap
from sqhsrc.algorithms.cc import cc_shap
from sqhsrc.algorithms.mcn import mcn
from sqhsrc.algorithms.cc_neyman import ccshap_neyman
from sqhsrc.utils.game_class import Game,DynamicGame
from sqhsrc.utils.tools import AverageErrorRatio,preprocess_adult,preprocess_wdbc,write_to_csv,preprocess_bank_marketing
import matplotlib.pyplot as plt
from sqhsrc.utils.options import args_parser
import math
import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


X_train,Y_train,X_test,Y_test = preprocess_bank_marketing()

model = SVC()
args = args_parser()
args.game = 'model'

args.ori_players = 500
args.new_players = 490
args.all_players = 500
args.samples = 500*args.new_players


game = Game(game_type=args.game,n=args.new_players,
    w=None,x_train= X_train, x_test= X_test, y_train = Y_train, y_test = Y_test, model = model)
# base_sv = cc_shap(game,10000*args.new_players,proc_num=4)

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
    #cc_sv = cc_shap(game,args.samples,proc_num=4)
    #neyman_sv = ccshap_neyman(game, 2, args.samples, proc_num=4) 
    mcn_sv = mcn(game,args.samples, proc_num=4) 
    dynamic_game = DynamicGame(game_type=args.game,n=args.all_players,
        dynamic_size=int(math.fabs(args.ori_players-args.new_players)),ori_shapley=ori_sv,ori_samples=50000*args.ori_players,
        w=None,x_train= X_train, x_test= X_test, y_train = Y_train, y_test = Y_test, model = model)
    #dynamic_sv = delete_delta_shap(dynamic_game,args.samples,proc_num=4)
    dynamic_neyman_sv = delete_delta_neyman_shap(dynamic_game,2,args.samples,proc_num=4)
    temp = []
    #temp.append(AverageErrorRatio(base_sv,cc_sv))
    #temp.append(AverageErrorRatio(base_sv,neyman_sv))
    #temp.append(AverageErrorRatio(base_sv,mcn_sv))
    #temp.append(AverageErrorRatio(base_sv,dynamic_sv))
    temp.append(AverageErrorRatio(base_sv,dynamic_neyman_sv))
    results.append(temp)

write_to_csv(results, './exp_results/bank_svm_delete_CCNeymanDelta.csv')
   












