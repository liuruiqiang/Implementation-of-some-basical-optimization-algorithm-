import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

class softmax_regression():
    def __init__(self, x_train, x_test, y_train, y_test, lr=0.01, epoches=200, batch_size=1,gamma = 0.9, eps=1e-8):
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
        self.gamma = gamma

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
        self.w = self.random_init((self.input_dim ,self.num_class))

        X,Y = x,y
        step = 0
        self.v = np.random.random(self.w.shape)
        batch_gen = self.batch_generator([X,Y])
        test_loss_list = []
        while(step < self.epoches):
            # print(self.A)
            j = np.random.randint(0,n)
            x,y = np.expand_dims(X[j],0),np.expand_dims(Y[j],0)
            pred_y = np.dot(x, self.w)
            test_acc, test_loss = self.test(self.x_test, self.y_test)
            test_loss_list.append(test_loss)
            loss = self.softmax_loss(pred_y, np.expand_dims(Y[j],0))
            grads = np.dot(x.T,(pred_y - np.expand_dims(Y[j],0)))
            #Nestrov梯度加速算法
            w_ = self.w + self.gamma * self.v
            pred_y = np.dot(x, w_)
            grads_ = np.dot(x.T, (pred_y - np.expand_dims(Y[j], 0)))
            self.v = self.gamma * self.v - self.lr * grads_
            self.w += self.v
            step += 1
        self.test_loss_list = test_loss_list

    def plot_loss(self, loss_list):
        iters = [i for i in range(self.epoches)]
        plt.plot(iters, loss_list)
        plt.title('loss in test', fontsize=24)
        plt.xlabel('iter', fontsize=4)
        plt.ylabel('loss', fontsize=4)
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
        print('labelshape', label.shape, pred.shape)
        return -np.mean(np.sum((np.log(pred) * label), axis=1))







