import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NCA():
    def __init__(self,low_dim,lr=0.01,epoches=200,batch_size=1,k_target=50):
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
        self.k_target = k_target

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

    def cost(self,x,y):
        (n, d) = x.shape
        A_x = np.dot(x, self.A)
        gradients = np.zeros((self.high_dim, self.high_dim))
        dist_matrix = np.sum((A_x[None, :, :] - A_x[:, None, :]) ** 2, axis=2)
        mask = (np.squeeze(y) == y)
        simlabel_matrix = mask * dist_matrix
        simlabel_matrix = np.where(simlabel_matrix == 0, np.inf, simlabel_matrix)
        sim_matrix = np.argsort(simlabel_matrix, axis=1)
        same_labels = np.ones((n, n))
        diffrent_labels = np.ones((n, n))
        for i in range(n):
            sum1 = np.zeros((self.high_dim, self.high_dim))
            sum2 = np.zeros((self.high_dim, self.high_dim))
            sum1 = 0
            sum2 = 0
            sum_lamda = 0
            for k in range(n):
                # 判断相似
                if k in list(sim_matrix[i,:self.k_target]) and same_labels[i, k] == 1 and same_labels[k, i] == 1:
                    # out_dot = np.outer(x[i] - x[k],x[i] - x[k])
                    x_i, x_k = np.expand_dims(x[i], axis=0), np.expand_dims(x[k], axis=0)
                    dot_result = np.dot(x_i - x_k, self.A)
                    dot_result = np.dot(dot_result, self.A.T)
                    dot_result = np.dot(dot_result, (x_i - x_k).T)
                    sum1 += dot_result
                    same_labels[k, i] = 0
                    same_labels[i, k] = 0
                #判断不相似
                if k not in list(sim_matrix[i,:self.k_target]) and diffrent_labels[i, k] == 1 and diffrent_labels[k, i] == 1:
                    x_i, x_k = np.expand_dims(x[i], axis=0), np.expand_dims(x[k], axis=0)
                    # dot_result = np.dot(x_i - x_k, (x_i - x_k).T)
                    dot_result2 = np.dot(x_i - x_k, self.A)
                    dot_result2 = np.dot(dot_result2, self.A.T)
                    dot_result2 = np.dot(dot_result2, (x_i - x_k).T)
                    sum2 += dot_result2
                    diffrent_labels[k, i] = 0
                    diffrent_labels[i, k] = 0
        # print('shape:',sum1.shape)
        return sum1[0,0] + self.lamda * (1 - sum2)

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
        A = self.random_init(A_shape)
        self.A = A
        self.lamda = np.random.uniform(0,5)
        A_x = np.dot(x, A)
        # 计算距离矩阵
        dist_matrix = np.sum((A_x[None, :, :] - A_x[:, None, :]) ** 2, axis=2)
        #mask表示是否属于同一类别，mask[i,j]==1表示i和j样本属于同一类别
        mask = (np.squeeze(y) == y)
        simlabel_matrix = mask * dist_matrix
        simlabel_matrix = np.where(simlabel_matrix == 0,np.inf,simlabel_matrix)
        #对样本，将与之属于同一类别的样本按照与它的距离排序
        sim_matrix = np.argsort(simlabel_matrix, axis=1)
        deltas_a = []
        costs = []
        step = 0
        while(step < self.epoches):
            # print(self.A)
            print('epoch:',step)
            gradients = np.zeros((self.high_dim, self.high_dim))
            same_labels = np.ones((n,n))
            diffrent_labels = np.ones((n,n))
            sum1 = 0
            sum2 = 0
            sum_lamda = 0
            for i in range(n):

                for k in range(n):
                    #判断相似，若在第i个样本的前k个同类样本列表中则相似
                    if k in list(sim_matrix[i,:self.k_target])  and same_labels[i,k] == 1 and same_labels[k,i] == 1:
                        # print('same labels')
                        # out_dot = np.outer(x[i] - x[k],x[i] - x[k])
                        x_i, x_k = np.expand_dims(x[i], axis=0), np.expand_dims(x[k], axis=0)
                        dot_result = np.dot(x_i - x_k, (x_i - x_k).T)
                        sum1 += dot_result
                        #避免重复计算
                        same_labels[k,i] = 0
                        same_labels[i,k] = 0
                    #判断不相似
                    if k not in list(sim_matrix[i,:self.k_target]) and diffrent_labels[i,k] == 1 and diffrent_labels[k,i] == 1:
                        # print('different labels')
                        x_i, x_k = np.expand_dims(x[i], axis=0), np.expand_dims(x[k], axis=0)
                        dot_result = np.dot(x_i - x_k, (x_i - x_k).T)
                        dot_result2 = np.dot(x_i - x_k,self.A)
                        dot_result2 = np.dot(dot_result2,self.A.T)
                        dot_result2 = np.dot(dot_result2,(x_i - x_k).T)
                        sum2 += dot_result
                        sum_lamda += dot_result2
                        #避免重复计算
                        diffrent_labels[k, i] = 0
                        diffrent_labels[i, k] = 0

            cost = self.cost(x,y)
            costs.append(cost)
            grads_a = sum1[0,0] *self.A + self.lamda * sum2 * self.A
            grads_lamda = 1 - sum_lamda
            gradients = -2 * np.dot(gradients,self.A)
            grads_anorm = np.linalg.norm(grads_a)
            print('delta:',grads_anorm)
            deltas_a.append(grads_anorm)
            #优化目标是最小化目标函数
            #梯度下降
            self.A -= self.lr * grads_a
            self.lamda -= self.lamda*grads_lamda
            step += 1
        self.deltas_a = deltas_a
        self.costs = costs


    def random_init(self,shape):
        A = np.random.uniform(size=shape)
        return A

    def plot_grad(self):
        iters = [i for i in range(self.epoches)]
        plt.plot(iters, self.deltas_a)
        plt.title('grads in training', fontsize=24)
        plt.xlabel('iter', fontsize=4)
        plt.ylabel('grads_A', fontsize=4)
        plt.show()

    def plot_cost(self):
        iters = [i for i in range(self.epoches)]
        plt.plot(iters, self.costs)
        plt.title('cost in training', fontsize=24)
        plt.xlabel('iter', fontsize=4)
        plt.ylabel('cost', fontsize=4)
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
    NCA_trainer = NCA(6,0.01,30,batch_size=10,k_target=50)
    NCA_trainer.train(X,Y)
    NCA_trainer.plot_grad()
    NCA_trainer.plot_cost()
    transformed_x = NCA_trainer.transform(X)
