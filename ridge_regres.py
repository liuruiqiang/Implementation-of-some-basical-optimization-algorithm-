import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split


def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = np.dot(xMat.T, xMat)
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    thata = np.dot(np.linalg.inv(denom),(np.dot(xMat.T,yMat)))
    y_hat = np.dot(xMat,thata)
    return thata,y_hat



if __name__ == "__main__":
    data = pd.read_csv('../abalone.data')
    uniqu_gender = np.unique(data['Sex'])
    #将字符型变量转换为数字
    data.loc[data['Sex']=='M','Sex'] = -1
    data.loc[data['Sex']=='F','Sex'] = 1
    data.loc[data['Sex']=='I','Sex'] = 0
    cols_name = data.columns
    x,y = np.array(data.iloc[:,:-1]),np.array(data.iloc[:,-1])
    y = np.expand_dims(y,1)
    x_mean = np.mean(x,0)

    x_var = np.var(x,0)
    y_mean = np.mean(y,0)
    # print(y_mean)
    # y = y - y_mean
    # x = (x - x_mean) / x_var
    #按照7:3随机划分训练集和测试集
    index = [i for i in range(len(y))]
    np.random.seed(0)
    np.random.shuffle(index)
    x,y = x[index],y[index]
    train_len = int(0.7 * len(y))
    x_train, x_test, y_train, y_test = x[:train_len],x[train_len:],y[:train_len],y[train_len:]
    print(x.shape,y.shape)
    losses = []
    #首先寻找最优lamda
    lamdas = np.linspace(0,5,50)
    for lam in lamdas:
        theta, y_hat = ridgeRegres(x_train, y_train,lam=lam)
        y_predicted = np.dot(x_test, theta)
        loss = np.mean(np.abs(y_predicted - y_test))
        losses.append(loss)
    # for i,(lamda,loss) in enumerate(zip(lamdas,losses)):
    #     print('lamda为{}时loss为{}'.format(lamda,loss))
    #     print('\n')
    plt.plot(lamdas,losses,'r',label='loss')
    plt.xlabel('lamda')
    plt.ylabel('loss')
    plt.show()
    losses = np.array(losses)
    index = np.argmin(losses)
    min_lamda = lamdas[index]
    min_loss = losses[index]
    print('最佳lamda为{},对应的损失为{}'.format(min_lamda, min_loss))
    #将最优lamda代入模型中得到最优模型并测试效果
    theta, y_hat = ridgeRegres(x_train, y_train,lam=min_lamda)
    y_predicted = np.dot(x_test, theta)
    loss = np.mean(np.abs(y_predicted - y_test))
    lens = len(y_test)
    y_predicted = np.dot(x_test, theta)
    plt.figure()
    plt.plot(range(lens), y_predicted, 'b', label="predict")
    plt.plot(range(lens), y_test, 'r', label="test")
    plt.legend(loc="upper right")
    plt.xlabel("number")
    plt.ylabel("Rings")
    plt.show()


