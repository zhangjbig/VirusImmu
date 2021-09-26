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

warnings.filterwarnings("ignore")


def descriptor_made_excel ():
    ###表示符表示
    # 参数
    descriptor_flag = 'E'
    # 路径
    test_path = 'data/test/test.csv'
    E_descriptor_path = 'data/E-descriptors.csv'
    test = pd.read_csv(test_path)
    E_descriptor = pd.read_csv(E_descriptor_path)


    if descriptor_flag == 'E':
        descriptor = E_descriptor
    else:
        pass

    for row in tqdm(test.itertuples()):
        id = row[1]
        content = row[3]
        if descriptor_flag == 'Z':
            result = pd.DataFrame(columns=['A', 'B', 'C'])
        elif descriptor_flag == 'E':
            result = pd.DataFrame(columns=['E1', 'E2', 'E3', 'E4', 'E5'])
        for i in content:
            if i in descriptor['name'].to_numpy():
                result = result.append(descriptor[descriptor['name'] == i].iloc[:, :], ignore_index=False)
            else:
                print('erro!', id, '_', i, )
        result.to_csv('data/test/descriptor_result/{}.csv'.format(id))

def descriptor_made_txt(content):

    data = pd.read_excel('data/E-descriptors.xlsx', header=0, index_col=0)
    result = pd.DataFrame(columns=['E1', 'E2', 'E3', 'E4', 'E5'])
    for index,item in enumerate(content):
        if item in data.index.values:
            result = result.append(data.loc[item], ignore_index=False)
        else:
            print('erro!', index , '_', item)
    if len(result) == len(content):
        pass
    else:
        print(content, ' is ERRO !', len(result), len(content))

    result.to_csv('data/test/descriptor_result/result.csv')


def ACC_caculated():
    # # # 读取描述符，计算acc，储存E_ACC.csv
    description = 'E'
    l = 8
    number = 10
    path = 'data/test/descriptor_result/result.csv'
    data = pd.read_csv(path, encoding='utf-8')
    n = len(data)
    if n < l:
        print('短了')
    else:
        result_flag = ACC.acc(path, number, l, description)
        result_flag.to_csv('data/test/acc_result/{0}_ACC_l={1}.csv'.format('E', l), encoding='utf-8')


class virus(object):

    def __init__(self,train_path_h,train_path_l,test_path):
        self.train_data_h = pd.read_csv(train_path_h, encoding='utf-8',index_col=0)
        self.train_data_l = pd.read_csv(train_path_l, encoding='utf-8',index_col=0)

        self.test_data = pd.read_csv(test_path, encoding='utf-8',index_col=0)

        self.train_data =  pd.concat([self.train_data_h, self.train_data_l], axis=0, ignore_index=True)

        self.x_train = self.train_data.drop(['name', 'label'], axis=1)
        self.y_train = self.train_data['label']

        self.x_test = self.test_data

        self.x_train_ = self.x_train
        self.x_test_ = self.x_test


    def choose_feature(self,flag):
        self.x_train_ = self.x_train[flag]
        self.x_test_ = self.x_test[flag]



    def predict_assemble(self,thred=0.4,random_j=1,importance=False,**parms):

        weight = [[0.05,0.75,0.2]]
        model_xg = self.mode_choose('xg')
        model_rf = self.mode_choose('rf')
        model_knn = self.mode_choose('knn')

        model_xg.fit(self.x_train_, self.y_train)
        model_rf.fit(self.x_train_, self.y_train)
        model_knn.fit(self.x_train_, self.y_train)
        self.y_p_xg = model_xg.predict_proba(self.x_test_)[:, 1]
        self.y_p_rf = model_rf.predict_proba(self.x_test_)[:, 1]
        self.y_p_knn = model_knn.predict_proba(self.x_test_)[:, 1]

        for item in weight:
            self.y_p_end = pd.Series((item[0]*self.y_p_xg) + (item[1]*self.y_p_rf) +(item[2]*self.y_p_knn))


            return self.y_p_end


    def predict(self,model_name = '',thred=0.4):
        record = pd.DataFrame()

        if model_name =='plsr':
            model = self.mode_choose(model_name)
            model.fit(self.x_train_, self.y_train)

            self.y_p_end = pd.Series(map(lambda x: x[0], model.predict(self.x_test_)))

            return self.y_p_end

        elif model_name == 'VirusImmu':


            return self.predict_assemble()


        else:

            model = self.mode_choose(model_name)
            model.fit(self.x_train_, self.y_train)
            self.y_p_end = pd.Series(model.predict_proba(self.x_test_)[:, 1])

            return self.y_p_end



    def mode_choose(self,model_name):

        if model_name == 'sgd':
            pass
        elif model_name == 'plsr':
            return PLSRegression(n_components=2)

        elif model_name == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        elif model_name == 'svm':
            return SVC(probability=True)
        elif model_name =='lr':
            return LogisticRegression()
        elif model_name == 'dt':
            return tree.DecisionTreeClassifier()
        elif model_name == 'xg':
            return  XGBClassifier()
        elif model_name == 'knn':
            # 最好的n_neighbors系数是2或者4
            return  KNeighborsClassifier(n_neighbors=2)
        elif model_name=='gbdt':
            return GradientBoostingClassifier(n_estimators=100)
        elif model_name =='ada':
            return AdaBoostClassifier(n_estimators=100)

        elif model_name == 'mlp':
            return MLPClassifier(hidden_layer_sizes=(32,16), solver='adam', activation='relu',max_iter=1000,
                                  random_state=42)

        elif model_name == 'all':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            svm = SVC(probability=True)
            gbdt = GradientBoostingClassifier(n_estimators=100)
            ada = AdaBoostClassifier(n_estimators=100)
            mlp = MLPClassifier(hidden_layer_sizes=(128, 64), solver='adam', activation='relu', max_iter=100,
                                  random_state=42,early_stopping=True)
            lr = LogisticRegression()
            dt = tree.DecisionTreeClassifier()
            knn = KNeighborsClassifier(n_neighbors=3)
            xg = XGBClassifier()

            return VotingClassifier(estimators=[('rf_', rf), ('xg_', xg),('knn_',knn)], voting='hard')











