import math
import os
import time
from pathlib import Path
from itertools import chain, combinations
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
import pandas as pd
from sklearn import metrics
from typing import Iterator
import csv

def compute_std(sv, n=False, exact=None):
    sv = np.copy(sv)
    if n:
        sv = [normalize(exact, svi) for svi in sv]

    mean = np.mean(sv, axis=0)
    # mean = [i + 0.1 for i in mean]
    r = np.std(sv, axis=0)

    c = 0
    cv = []
    for j in range(len(mean)):
        if mean[j] != 0:
            c += 1
            # if abs(r[j] / mean[j]) < 0.5:
            cv.append(abs(r[j] / mean[j]))
    print('not zero:', c)
    if c == 0:
        return math.inf
    else:
        mr = np.mean(cv)
        ar = np.max(cv)
    return mr


def eval_utility(x_train, y_train, x_test, y_test, model) -> float:
    """Evaluate the coalition utility.
    """
    single_pred_label = (True if len(np.unique(y_train)) == 1 else False)

    if single_pred_label:
        y_pred = [y_train[0]] * len(y_test)
    else:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

    return metrics.accuracy_score(y_test, y_pred, normalize=True)

def AverageErrorRatio(benchmark,estimate):
    sum = 0
    for i in range(len(benchmark)):
        sum += math.fabs((benchmark[i]-estimate[i])/benchmark[i])
    sum /= len(benchmark)
    return sum


def eval_game_utility(x, gt, n=1, w=None):
    """Evaluate the coalition utility.
    """
    if len(x) == 0:
        return 0
    if gt == 'voting':
        if w is None:
            w = [
                45, 41, 27, 26, 26, 25, 21, 17, 17, 14, 13, 13, 12, 12, 12, 11,
                10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 5, 4,
                4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3
            ]
        else:
            w = load_npy(w)
        m = sum(w) / 2
        r = np.sum(np.array(w)[x])
        return 1 if r > m else 0
    elif gt == 'airport':
        if w is None:
            w = [
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
                4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8,
                8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9
            ]
        else:
            w = load_npy(w)
        r = np.max(np.array(w)[x])
        return r
    elif gt == 'tree':
        r = n
        x = sorted(x)
        m = r - (x[-1] - x[0])
        for i in range(1, len(x)):
            t = x[i] - x[i - 1]
            if t > m:
                m = t
        return r - m


def power_set(iterable) -> Iterator:
    """Generate the power set of the all elements of an iterable obj.
    """
    s = list(iterable)
    return chain.from_iterable(
        combinations(s, r) for r in range(1,
                                          len(s) + 1))

def time_function(f, *args) -> float:
    """Call a function f with args and return the time (in seconds)
    that it took to execute.
    """

    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


def get_ele_idxs(ele, ele_list) -> list:
    """Return all index of a specific element in the element list
    """
    idx = -1
    if not isinstance(ele_list, list):
        ele_list = list(ele_list)
    n = ele_list.count(ele)
    idxs = [0] * n
    for i in range(n):
        idx = ele_list.index(ele, idx + 1, len(ele_list))
        idxs[i] = idx
    return idxs


def split_num(m_list, num) -> np.ndarray:
    """Split num
    """
    perm_arr_list = None

    for m in m_list:
        assert m >= 0
        if m != 0:
            m = int(m)
            quotient = int(m / num)
            remainder = m % num
            if remainder > 0:
                perm_arr = [[quotient]] * (num - remainder) + [[quotient + 1]
                                                               ] * remainder
            else:
                perm_arr = [[quotient]] * num
        else:
            perm_arr = [[0]] * num
        if perm_arr_list is None:
            perm_arr_list = perm_arr
        else:
            perm_arr_list = np.concatenate((perm_arr_list, perm_arr), axis=-1)

    return np.asarray(perm_arr_list)


def split_permutation(m, num) -> np.ndarray:
    """Split permutation
    """
    assert m > 0
    quotient = int(m / num)
    remainder = m % num

    perm_arr = []
    r = []
    for i in range(m):
        r.append(i)
        if (remainder > 0
                and len(r) == quotient + 1) or (remainder <= 0
                                                and len(r) == quotient):
            remainder -= 1
            perm_arr.append(r)
            r = []
    return perm_arr


def split_permutation_num(m, num) -> np.ndarray:
    """Split permutation num
    """
    assert m > 0
    quotient = int(m / num)
    remainder = m % num
    if remainder > 0:
        perm_arr = [quotient] * (num - remainder) + [quotient + 1] * remainder
    else:
        perm_arr = [quotient] * num
    return np.asarray(perm_arr)


def split_permutations_t_list(permutations, t_list, num) -> list:
    """Split permutation num
    """
    m = len(permutations)
    m_list = split_permutation_num(m, num)
    res = list()
    for local_m in m_list:
        res.append([permutations[:local_m], t_list[:local_m]])
        permutations = permutations[local_m:]
        t_list = t_list[local_m:]
    return res


def save_npy(file_name, arr):
    check_folder('res')
    if (isinstance(arr, np.ndarray)):
        np.save(Path.cwd().joinpath('res').joinpath(file_name), arr)


def load_npy(file_name):
    check_folder('res')
    arr = np.load(Path.cwd().joinpath('res').joinpath(file_name))
    if (isinstance(arr, np.ndarray)):
        return arr


def check_folder(floder_name):
    """Check whether the folder exists
    """
    if Path(floder_name).exists() == False:
        prefix = Path.cwd()
        Path.mkdir(prefix.joinpath(floder_name))


def normalize(list1, list2):
    """Rormalize list1 to list2
    """
    coef = np.sum(list1) / np.sum(list2)
    return coef * list2


def power_set(iterable):
    s = list(iterable)
    return chain.from_iterable(
        combinations(s, r) for r in range(1,
                                          len(s) + 1))


def preprocess_data_sub(train_file_name, valid_file_name):
    """Process the training dataset and the valid dataset
    """
    train_df = pd.read_csv(
        os.path.abspath('.') + '/data_files/' + train_file_name)
    valid_df = pd.read_csv(
        os.path.abspath('.') + '/data_files/' + valid_file_name)

    # train_df = pd.read_csv(r'../data_files/' + train_file_name)
    # valid_df = pd.read_csv(r'../data_files/' + valid_file_name)

    columns_name = train_df.columns

    x_train = train_df.drop(columns=['Y']).values
    x_valid = valid_df.drop(columns=['Y']).values
    y_train = train_df.Y.values
    y_valid = valid_df.Y.values

    return x_train, y_train, x_valid, y_valid, columns_name

def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    print(f"Data has been written to {filename} successfully.")

def preprocess_wdbc():
    # 读取数据文件
    data_path = './data_sets/breast-cancer-wisconsin.data'
    column_names = ['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses', 'Class']
    df = pd.read_csv(data_path, names=column_names)

    # 删除存在缺失值的特征列
    df.dropna(subset=['Bare Nuclei'], inplace=True)

    # 划分数据集
    train_data = df[:600]  # 前600条数据作为训练集
    test_data = df[600:]   # 后99条数据作为测试集

    # 提取特征和标签
    X_train = train_data.drop(['ID', 'Clump Thickness','Uniformity of Cell Size', 'Class', 'Bare Nuclei','Bland Chromatin',
                'Normal Nucleoli', 'Mitoses'], axis=1).astype(int)
    Y_train = train_data['Class'].astype(int)

    X_test = test_data.drop(['ID','Clump Thickness', 'Uniformity of Cell Size','Class', 'Bare Nuclei','Bland Chromatin',
                'Normal Nucleoli', 'Mitoses'], axis=1).astype(int)
    Y_test = test_data['Class'].astype(int)

    X_train = X_train.to_numpy()
    Y_train = Y_train.to_numpy()
    X_test = X_test.to_numpy()
    Y_test = Y_test.to_numpy()
    return X_train,Y_train,X_test,Y_test

def preprocess_adult():
    # 读取数据集
    data = pd.read_csv('./data_sets/adult.data', header=None)

    # 命名特征列
    columns = ['Age', 'Workclass', 'Final Weight', 'Education', 'Education Number', 'Marital Status',
           'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss',
           'Hours per Week', 'Native Country', 'Income']

    # 替换缺失值为NaN
    data = data.replace('?', np.nan)

    # 删除包含缺失值的行
    data = data.dropna()

    # 重新设置列名
    data.columns = columns

    # 选择没有缺失值的特征列
    features = data.drop(['Workclass','Final Weight',  'Education', 'Education Number', 'Marital Status', 'Occupation', 'Relationship',
                      'Race', 'Sex', 'Native Country', 'Income'], axis=1)

    # 划分特征和标签
    X = features.to_numpy()
    Y = data['Income'].to_numpy()

    # 划分训练集和验证集
    X_train = X[:600]
    Y_train = Y[:600]
    X_test = X[600:800]
    Y_test = Y[600:800]

    return X_train,Y_train,X_test,Y_test

def preprocess_bank_marketing():
    # 读取数据集
    data_path = './data_sets/bank.csv' # 路径需根据实际情况更改
    data = pd.read_csv(data_path, sep=';') # 分隔符根据实际文件更改

    # 命名特征列（如果文件没有列名）
    columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

    # 如果文件没有列名，取消注释下一行
    # data.columns = columns

    # 选择分类特征进行独热编码
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    encoder = OneHotEncoder(drop='first')
    encoded_features = encoder.fit_transform(data[categorical_features])

    # 替换原始分类特征列
    data = data.drop(columns=categorical_features)
    data = pd.concat([data, pd.DataFrame(encoded_features.toarray())], axis=1)

    # 划分特征和标签
    X = data.drop(['y'], axis=1).to_numpy() # 'y'是标签列，需根据实际情况更改
    Y = data['y'].to_numpy()

    # 划分训练集和测试集
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    X_train = X[:600]
    Y_train = Y[:600]
    X_test = X[600:800]
    Y_test = Y[600:800]

    return X_train, Y_train, X_test, Y_test