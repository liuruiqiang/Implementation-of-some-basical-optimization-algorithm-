import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_rcv1
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets

class softmax_regression():
    def __init__(self, x_train, x_test, y_train, y_test, lr=0.01, epoches=200, batch_size=1,alpha=0.9, beta=0.999, eps=1e-8):
        """
        :param low_dim:lower dimension of A
        :param lr:learning rate
        :param epoches:the steps of trainning
        :param batch_size:size of a batch
        """

        self.lr = lr
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        self.epoches = epoches
        self.batch_size = batch_size
        self.s = 0
        self.r = eps
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.alpha_i = 1
        self.beta_i = 1

    def shuffle_data(self,data):
        """
        random shuffle the data
        :param data: [x,y]
        :return:
        """
        n_sample = data[0].shape[0]
        index = np.random.permutation(n_sample)
        return [d[index] for d in data]

    def batch_generator(self,data,shuffle=True):
        """
        yield batch of data
        :param data: [x,y]
        :return:
        """
        batch_count = 0
        if shuffle:
            data = self.shuffle_data(data)
        while True:
            if batch_count * self.batch_size + self.batch_size > len(data[0]):
                batch_count = 0
            start = batch_count * self.batch_size
            end = batch_count * self.batch_size + self.batch_size
            batch_count += 1
            yield [d[start:end] for d in data]

    def update(self,grads):
        '''
        :param grads:
        :return: Nadamx对梯度改进后得到的参数修改项
        '''
        self.s = self.s * self.alpha + (1 - self.alpha) * grads
        self.r = np.maximum(self.r * self.beta, np.abs(grads))
        self.alpha_i *= self.alpha
        lr = -self.lr / (1 - self.alpha_i)
        return lr * (self.s * self.alpha + (1 - self.alpha) * grads) / self.r


    def train_on_batch(self,x,y):
        """

        :param x:data,(n,d)
        :param y:label,(n,numclass)
        :return:
        """
        (n, d) = x.shape
        (n,num_class) = y.shape
        x = np.column_stack((x, np.ones((n, 1))))
        (n, d) = x.shape
        self.input_dim ,self.output_dim = d , num_class
        self.w = self.random_init((self.input_dim ,self.output_dim))

        X,Y = x,y
        step = 0
        batch_gen = self.batch_generator([X,Y])
        test_loss_list = []
        test_acc_list = []
        Normdelta_list = []
        test_error_list = []
        min_loss = 10000
        self.s = np.zeros_like(self.w)
        self.r = np.ones_like(self.w) * self.r
        while(step < self.epoches):
            # print(self.A)
            j = np.random.randint(0,n)
            x,y = np.expand_dims(X[j],0),np.expand_dims(Y[j],0)
            pred_y = np.dot(x, self.w)
            test_acc, test_loss = self.test(self.x_test, self.y_test)
            if test_loss < min_loss:
                min_loss = test_loss
            test_error = 1 - test_acc
            test_error_list.append(test_error)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            grads = np.dot(x.T,(pred_y - np.expand_dims(Y[j],0)))
            delta_w = self.update(grads)
            norm_delta = np.linalg.norm(delta_w)
            Normdelta_list.append(norm_delta)
            self.w += delta_w
            step += 1
        self.test_loss_list = test_loss_list - min_loss
        self.Normdelta_list = Normdelta_list
        self.test_error_list = test_error_list

    def plot_loss(self, loss_list):
        iters = [i for i in range(self.epoches)]
        plt.plot(iters, loss_list)
        plt.title('gap between loss and  min_loss', fontsize=24)
        plt.xlabel('iter', fontsize=4)
        plt.ylabel('difference', fontsize=4)
        plt.show()

    def plot_Normdelta(self):
        iters = [i for i in range(self.epoches)]
        plt.plot(iters, self.Normdelta_list)
        plt.title('length of the change', fontsize=24)
        plt.xlabel('iter', fontsize=4)
        plt.ylabel('length(2 norm)', fontsize=4)
        plt.show()

    def plot_test_errors(self, error_list):
        iters = [i for i in range(self.epoches)]
        plt.plot(iters, error_list)
        plt.title('errors in test', fontsize=24)
        plt.xlabel('iter', fontsize=4)
        plt.ylabel('error', fontsize=4)
        plt.show()

    def test(self, x, y):
        (n, d) = x.shape
        (n, num_class) = y.shape
        x = np.column_stack((x, np.ones((n, 1))))  # 增加偏置项
        pred_y = np.dot(x, self.w)
        pred_y = self.softmax(pred_y)
        test_loss = self.softmax_loss(pred_y, y)
        pred = pred_y.argmax(axis=1)
        y = y.argmax(axis=1)
        result = [pred[i] == y[i] for i in range(len(pred))]

        acc = sum(result) / len(result)

        return acc, test_loss

    def softmax(self, y):
        exp_pred = np.exp(y)
        exp_predsum = np.expand_dims(np.sum(exp_pred, axis=1), 1)
        # print('exppredshape:',exp_pred.shape,exp_predsum.shape)
        pred = exp_pred / exp_predsum
        return pred

    def random_init(self, shape):
        n_features, num_class = shape
        limit = np.sqrt(1 / n_features)
        W = np.random.uniform(-limit, limit, shape)
        return W

    def softmax_loss(self, pred, label):
        '''
        calculate the loss between pred and label
        :param pred:
        :return:
        '''
        # print('labelshape', label.shape, pred.shape)
        return -np.mean(np.sum((np.log(pred) * label), axis=1))


if __name__ == "__main__":
    data_path = "../covtype.data"
    data = pd.read_csv(data_path,header=None)
    qualitative_list = []
    # 统计零一变量的特征
    for i in range(54):
        # print(np.unique(data.iloc[:,i]))
        if len(np.unique(data.iloc[:,i])) == 2:
            qualitative_list.append(i)
    print(qualitative_list)

    X ,Y = np.array(data.iloc[:,:-1]),np.array(data.iloc[:,-1])
    Y = np.expand_dims(Y, 1)
    x_mean = np.mean(X[:,:10],axis=0)
    x_var = np.var(X[:,:10],axis=0)
    #对非零一变量特征进行标准化处理
    X[:,:10] = (X[:,:10] - x_mean)/x_var
    (n, d) = X.shape
    train_len = int(0.7 * n)
    index = [i for i in range(n)]
    np.random.seed(0)
    np.random.shuffle(index)
    enc = OneHotEncoder(sparse=False)
    Y = enc.fit_transform(Y)
    X,Y = X[index],Y[index]
    # 按照3:7的比例对数据集划分为训练集和测试集
    x_train,x_test,y_train,y_test = X[:train_len],X[train_len:],Y[:train_len],Y[train_len:]
    model = softmax_regression(x_train,x_test,y_train,y_test,lr=0.01,batch_size=1,epoches=1000)
    model.train_on_batch(x_train,y_train)
    acc,loss = model.test(x_test,y_test)
    model.plot_loss(model.test_loss_list)
    model.plot_Normdelta()
    model.plot_test_errors(model.test_error_list)








