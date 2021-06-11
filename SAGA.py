import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class softmax_regression():
    def __init__(self, x_train, x_test, y_train, y_test, lr=0.01, epoches=200, batch_size=1):
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
        batch_gen = self.batch_generator([X,Y])

        saved_grads = np.zeros((n,self.input_dim ,self.num_class))
        test_loss_list = []
        for i in range(n):
            pred_y = np.dot(np.expand_dims(x[i],axis=0), self.w)
            # loss = self.softmax_loss(pred_y, y[i])
            grads = np.dot(x[i].T, (pred_y - np.expand_dims(y[i],axis=0)))
            saved_grads[i] = grads
        while(step < self.epoches):
            # print(self.A)
            j = np.random.randint(0,n)
            x,y = np.expand_dims(X[j],0),np.expand_dims(Y[j],0)
            pred_y = np.dot(x, self.w)
            test_acc, test_loss = self.test(self.x_test, self.y_test)
            test_loss_list.append(test_loss)
            loss = self.softmax_loss(pred_y, np.expand_dims(Y[j],0))
            grads = np.dot(x.T,(pred_y - np.expand_dims(Y[j],0)))
            d = grads - saved_grads[j] + np.mean(saved_grads,axis=0)
            saved_grads[j] = grads
            self.w -= self.lr * d
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


if __name__ == "__main__":
    data_path = "../wine.data"
    data = pd.read_csv(data_path)
    # names = pd.read_csv('../wine.names')
    print(data)
    print(data.iloc[:,0])

    X ,Y = np.array(data.iloc[:,1:14]),np.array(data.iloc[:,0])
    x_mean = np.mean(X,axis=0)
    x_var = np.var(X,axis=0)
    X = (X - x_mean)/x_var
    (n, d) = X.shape
    # x = np.insert(X, 13, np.ones((n, )), axis=1)
    X = np.column_stack((X,np.ones((n,1))))
    print(X.shape)
    print(X[0].shape)
    # Y = np.expand_dims(Y,1)
    # NCA_trainer = NCA(6,0.01,200,batch_size=10)
    # NCA_trainer.train_on_batch(X,Y)
    # transformed_x = NCA_trainer.transform(X)
    # print(transformed_x)
    # print(NCA_trainer.A)
