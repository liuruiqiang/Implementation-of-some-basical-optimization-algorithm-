import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NCA():
    def __init__(self,low_dim,lr=0.01,epoches=200,batch_size=1):
        """

        :param low_dim:lower dimension of A
        :param lr:learning rate
        :param epoches:the steps of trainning
        :param batch_size:size of a batch
        """
        self.low_dim = low_dim
        self.lr = lr
        self.epoches = epoches
        self.batch_size = batch_size

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

    def train(self,x,y):
        """
        :param x:data,(n,d)
        :param y:label,(n,1)
        :return:
        """
        (n, d) = x.shape
        high_dim = d
        self.high_dim = d
        A_shape = (self.high_dim,self.low_dim)
        #随机初始化参数
        A = self.random_init(A_shape)
        self.A = A
        step = 0
        while(step < self.epoches):
            # print(self.A)
            A_x = np.dot(x, A)
            # 计算距离矩阵
            dist_matrix = np.sum((A_x[None, :, :] - A_x[:, None, :]) ** 2, axis=2)
            # 转换为e的指数形式
            exp_mat = np.exp(-dist_matrix)
            # 对角线元素置为0，减去一个只包含对角线元素的矩阵
            exp_mat = exp_mat - np.diag(np.diag(exp_mat))
            # print('expmat',exp_mat)
            # 得到概率矩阵
            prob_mat = exp_mat / np.sum(exp_mat, axis=1).reshape((-1, 1))
            # print('probmat', exp_mat)
            # 得到pi
            # print('promat_shape',prob_mat.shape)
            prob_i = np.array([np.sum(prob_mat[i][y[i] == y]) for i in range(n)])
            gradients = np.zeros((self.high_dim, self.high_dim))
            for i in range(n):
                sum1 = np.zeros((self.high_dim,self.high_dim))
                sum2 = np.zeros((self.high_dim,self.high_dim))
                for k in range(n):
                    #做外积
                    out_dot = np.outer(x[i] - x[k],x[i] - x[k])
                    sum1 += prob_mat[i,k] * out_dot
                    if y[i] == y[k]:
                        sum2 += prob_mat[i,k] * out_dot
                sum1 = prob_i[i] * sum1
                gradients += sum1 - sum2
            gradients = -2 * np.dot(gradients,self.A)
            grad_norm = np.linalg.norm(gradients)
            #优化目标是最小化目标函数
            self.A -= self.lr * gradients
            step += 1

    def train_on_batch(self,x,y):
        """
        :param x:data,(n,d)
        :param y:label,(n,1)
        :return:
        """
        (n, d) = x.shape
        high_dim = d
        self.high_dim = d
        A_shape = (self.high_dim,self.low_dim)
        A = self.random_init(A_shape)
        self.A = A
        X,Y = x,y
        step = 0
        #随机产生minibatch的样本
        batch_gen = self.batch_generator([X,Y])
        delta = 0
        deltas = []
        while(step < self.epoches):
            # print(self.A)
            batch_x,batch_y = next(batch_gen)
            # print('batch',batch_x.shape,batch_y.shape)
            x,y = batch_x,batch_y
            A_x = np.dot(x, A)
            # 计算距离矩阵
            dist_matrix = np.sum((A_x[None, :, :] - A_x[:, None, :]) ** 2, axis=2)
            # 转换为e的指数形式
            exp_mat = np.exp(0.0 -dist_matrix)
            # 对角线元素置为0，减去一个只包含对角线元素的矩阵
            exp_mat = exp_mat - np.diag(np.diag(exp_mat))
            # print('expmat',exp_mat)
            # 得到概率矩阵
            prob_mat = exp_mat / np.sum(exp_mat, axis=1).reshape((-1, 1))
            # 得到pi
            prob_i = np.array([np.sum(prob_mat[i][y == y[i]]) for i in range(self.batch_size)])
            gradients = np.zeros((self.high_dim, self.high_dim))
            for i in range(self.batch_size):
                sum1 = np.zeros((self.high_dim,self.high_dim))
                sum2 = np.zeros((self.high_dim,self.high_dim))

                for k in range(self.batch_size):
                    #做外积
                    out_dot = np.outer(x[i] - x[k],x[i] - x[k])
                    sum1 += prob_mat[i,k] * out_dot
                    if y[i] == y[k]:
                        sum2 += prob_mat[i,k] * out_dot
                sum1 = prob_i[i] * sum1
                gradients += sum1 - sum2
            #得到最后的梯度
            gradients = -2 * np.dot(gradients,self.A)
            delta = np.linalg.norm(gradients)
            deltas.append(delta)
            print("梯度范数为：",delta)
            #优化目标是最小化目标函数
            #梯度下降更新
            self.A -= self.lr * gradients
            step += 1
        self.deltas = deltas

    def random_init(self,shape):
        A = np.random.uniform(size=shape)
        return A

    def plot_grad(self):
        iters = [i for i in range(self.epoches)]
        plt.plot(iters, self.deltas)
        plt.title('grads in training', fontsize=24)
        plt.xlabel('iter', fontsize=4)
        plt.ylabel('grads', fontsize=4)
        plt.show()

    def transform(self,x):
        '''
        transform x into low dimension space using trained A
        :param x:
        :return:
        '''
        return np.dot(x,self.A)

if __name__ == "__main__":
    data_path = "../wine.data"
    data = pd.read_csv(data_path)
    X ,Y = np.array(data.iloc[:,1:14]),np.array(data.iloc[:,0])
    #标准化
    x_mean = np.mean(X,axis=0)
    x_var = np.var(X,axis=0)
    X = (X - x_mean)/x_var
    # Y = np.expand_dims(Y,1)
    NCA_trainer = NCA(6,0.01,200,batch_size=10)
    NCA_trainer.train_on_batch(X,Y)
    NCA_trainer.plot_grad()
    transformed_x = NCA_trainer.transform(X)

