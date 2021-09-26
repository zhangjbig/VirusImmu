"""
Created on 21st Sep 25 00:00:09 2020
@author: JL,JZ, email: lijingj1006@163.com
"""

import pandas as pd
import os
import ACC
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,roc_curve,auc
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
from random import sample
from sklearn.cross_decomposition import PLSRegression
import Model
warnings.filterwarnings("ignore")


def main():

    description = 'E'
    l = 8
    thred = 0.4

    train_path_h = 'data/train/protect/{}_ACC_l={}.csv'.format(description, l)
    train_path_l = 'data/train/unprotect/{}_ACC_l={}.csv'.format(description, l)

    test_path = 'data/test/acc_result/{}_ACC_l={}.csv'.format(description, l)


    with open('data/train/{}-l={}-rf-importances'.format(description, l), 'rb') as f:
        x_label = pickle.load(f)

    virus_1 = Model.virus(train_path_h, train_path_l, test_path)

    for i in [22]:
        x_label_result_true = [i[0] for i in x_label[:i]]

        virus_1.choose_feature(x_label_result_true)
        model_name  = 'VirusImmu'
        result = virus_1.predict(model_name=model_name, thred=thred)
        print(result.values)
        result.to_csv('output/result.csv',mode='w',encoding='utf-8')

if __name__ == '__main__':

    content = 'TGRIRVLSF'

    # Model.descriptor_made_txt(content)

    # Model.ACC_caculated()

    main()




