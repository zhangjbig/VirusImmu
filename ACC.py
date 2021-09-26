import pandas as pd
import numpy as np

#acc降维过程
def acc(path,number,l,description):
    data  = pd.read_csv(path,encoding='utf-8')
    n = len(data)
    acc_result = pd.Series()


    if description =='Z':
        a_list = data['A']
        b_list = data['B']
        c_list = data['C']
        acc_jisuan_Z(n, a_list, b_list, c_list,acc_result, number,l)

    elif description =='E':
        a_list = data['E1']
        b_list = data['E2']
        c_list = data['E3']
        d_list = data['E4']
        e_list = data['E5']
        acc_jisuan_E(n,a_list,b_list,c_list,d_list,e_list,acc_result,number,l)

    acc_result_ = pd.DataFrame()
    acc_result_ = acc_result_.append(acc_result,ignore_index=True)
    return  acc_result_

def sigmod_(number,flag):
    if flag == 0:
        return number
    elif flag == 1:
        # return  1/(1 + np.exp(-number)
        return np.exp(number)
    elif flag == 2:
        # return 1 / (1 + np.exp(-(2*number)))
        return np.exp(2*number)
    elif flag ==3:
        # return 1 / (1 + np.exp(-(0.5*number))
        return np.exp(0.5 * number)

# acc降维计算具体函数1
def acc_jisuan_Z(n, data1, data2, data3, result, number, l):

    l_list = np.arange(1,l+1)
    for l in l_list:
        x11 = round(sum(((data1[:n - l] * data1[1:n - l + 1].tolist()) / (n - l))), number)
        result['A{0}{1}({2})'.format(1, 1, l)] = x11
        x12 = round(sum(((data1[:n - l] * data2[1:n - l + 1].tolist()) / (n - l))), number)
        result['A{0}{1}({2})'.format(1, 2, l)] = x12
        x13 = round(sum(((data1[:n - l] * data3[1:n - l + 1].tolist()) / (n - l))), number)
        result['A{0}{1}({2})'.format(1, 3, l)] = x13


        x21 = round(sum((data2[:n - l] * data1[1:n - l + 1].tolist()) / (n - l)), number)
        result['A{0}{1}({2})'.format(2, 1, l)] = x21
        x22 = round(sum((data2[:n - l] * data2[1:n - l + 1].tolist()) / (n - l)), number)
        result['A{0}{1}({2})'.format(2, 2, l)] = x22
        x23 = round(sum((data2[:n - l] * data3[1:n - l + 1].tolist()) / (n - l)), number)
        result['A{0}{1}({2})'.format(2, 3, l)] = x23

        #
        x31 = round(sum((data3[:n - l] * data1[1:n - l + 1].tolist()) / (n - l)), number)
        result['A{0}{1}({2})'.format(3, 1, l)] = x31
        x32 = round(sum((data3[:n - l] * data2[1:n - l + 1].tolist()) / (n - l)), number)
        result['A{0}{1}({2})'.format(3, 2, l)] = x32
        x33 = round(sum((data3[:n - l] * data3[1:n - l + 1].tolist()) / (n - l)), number)
        result['A{0}{1}({2})'.format(3, 3, l)] = x33


def acc_jisuan_E(n,data1,data2,data3,data4,data5,result,number,l):

    l_list = np.arange(1,l+1)
    for l in l_list:

        x11 = round(sum(((data1[:n-l]*data1[1:n-l+1].tolist())/(n-l))),number)
        result['A{0}{1}({2})'.format(1,1,l)] = x11
        x12 = round(sum(((data1[:n-l]*data2[1:n-l+1].tolist())/(n-l))),number)
        result['A{0}{1}({2})'.format(1,2,l)] = x12
        x13 = round(sum(((data1[:n-l]*data3[1:n-l+1].tolist())/(n-l))),number)
        result['A{0}{1}({2})'.format(1, 3, l)] = x13
        x14 = round(sum(((data1[:n - l] * data4[1:n - l + 1].tolist()) / (n - l))),number)
        result['A{0}{1}({2})'.format(1, 4, l)] = x14
        x15 = round(sum(((data1[:n - l] * data5[1:n - l + 1].tolist()) / (n - l))),number)
        result['A{0}{1}({2})'.format(1, 5, l)] = x15

        x21 = round(sum((data2[:n-l]*data1[1:n-l+1].tolist())/(n-l)),number)
        result['A{0}{1}({2})'.format(2,1,l)] = x21
        x22 = round(sum((data2[:n-l]*data2[1:n-l+1].tolist())/(n-l)),number)
        result['A{0}{1}({2})'.format(2,2,l)] = x22
        x23 = round(sum((data2[:n - l] * data3[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(2, 3, l)] = x23
        x24 = round(sum((data2[:n - l] * data4[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(2, 4, l)] = x24
        x25 = round(sum((data2[:n - l] * data5[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(2, 5, l)] = x25
        #
        x31 =round( sum((data3[:n-l]*data1[1:n-l+1].tolist())/(n-l)),number)
        result['A{0}{1}({2})'.format(3,1,l)] = x31
        x32 = round(sum((data3[:n-l]*data2[1:n-l+1].tolist())/(n-l)),number)
        result['A{0}{1}({2})'.format(3,2,l)] = x32
        x33 = round(sum((data3[:n - l] * data3[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(3, 3, l)] = x33
        x34 = round(sum((data3[:n - l] * data4[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(3, 4, l)] = x34
        x35 = round(sum((data3[:n - l] * data5[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(3, 5, l)] = x35

        x41 = round(sum((data4[:n - l] * data1[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(4, 1, l)] = x41
        x42 = round(sum((data4[:n - l] * data2[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(4, 2, l)] = x42
        x43 = round(sum((data4[:n - l] * data3[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(4, 3, l)] = x43
        x44 = round(sum((data4[:n - l] * data4[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(4, 4, l)] = x44
        x45 = round(sum((data4[:n - l] * data5[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(4, 5, l)] = x45

        x51 = round(sum((data5[:n - l] * data1[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(5, 1, l)] = x51
        x52 = round(sum((data5[:n - l] * data2[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(5, 2, l)] = x52
        x53 =round(sum((data5[:n - l] * data3[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(5, 3, l)] = x53
        x54 = round(sum((data5[:n - l] * data4[1:n - l + 1].tolist()) / (n - l)),number)
        result['A{0}{1}({2})'.format(5, 4, l)] = x54
        x55 = round(sum((data5[:n - l] * data5[1:n - l + 1].tolist()) / (n - l)),number)
        result.loc['A{0}{1}({2})'.format(5, 5, l)] = x55

