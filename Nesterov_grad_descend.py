import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

class optimizer(object):
    def __init__(self,f_mode='f',rph=0.1,init_alpha=1,t=2,gamma = 0.9,lr=0.1):
        self.f_mode = f_mode
        self.rph = rph
        self.init_alpha = init_alpha
        self.t = t
        self.gamma = gamma
        self.lr = lr
    
    def search_alpha(self,x,d,rph,init_alpha,coe):
        '''
        针对第四题这种函数已知的情况利用Goldstein原则对alpha进行非精确线性搜索
        '''

        flag = False
        a = 0
        b = init_alpha
        fx = self.f(x)
        gx = self.g(x)
        f0 = fx
        
        gd0 = np.dot(gx,d)
        # print("gx",gx,d,gd0)
        alpha = b * np.random.uniform(0,1)
        while(not flag):
            newfx = self.f(x + alpha *d)
            fi = newfx
            if (fi <= f0 + rph *alpha * gd0):
                if (fi >= f0 + (1-rph) * alpha *gd0):
                    flag = True
                else:
                    a = alpha
                    b = b
                    if (b < init_alpha):
                        alpha = (a +b)/2
                    
                    else:
                        alpha = coe * alpha
            else:
                a = a
                b = alpha
                alpha = (a +b)/2

        return alpha

    def f(self,x):
        '''
        返回第四题函数数值结果
        '''
        x1 ,x2 = x[0],x[1]
        return 0.1*x1**2 + 2*x2**2

    def g(self,x):
        '''
        返回第四题函数梯度数值结果
        '''
        x1 ,x2 = x[0],x[1]
        g1 = 0.2*x1
        g2 = 4*x2
        results = np.array([g1,g2])
        return results

    def gradient_descend(self,x0,epoches=1500,eps=1e-6):
        '''
        利用最速下降法进行优化的过程
        '''
        fx0 = self.f(x0)
        grad0 = self.g(x0)
        lens = len(x0)
        delta = np.sum(grad0**2)
        i = 1
        x = x0
        record_x = np.zeros((lens, epoches))  # 存储优化过程x的更新值
        record_x[:, 0] = x0
        record_f = np.zeros((1,epoches))
        record_g = np.zeros((lens,epoches))
        record_f[0,0] = fx0
        record_g[:,0] = grad0
        v = np.random.random(grad0.shape)
        while(i < epoches and delta >eps):
            x_ = x + self.gamma * v
            g_ = self.g(x_)
            v = self.gamma * v - self.lr * g_
            x = x + v
            record_x[:, i] = x
            grad = self.g(x)
            # print('grad:',grad)
            fx = self.f(x)
            record_f[0,i] = fx
            record_g[:,i] = grad
            delta = np.sum(grad **2)
            i += 1

        return x,i,record_f[:,:i],record_g[:,:i],record_x[:,:i]

    def __call__(self,x0):
        final_x,epoches,records_f,records_g,record_x = self.gradient_descend(x0)
        return final_x,epoches,records_f,records_g,record_x

def muti_sovler(lr,gamma):
    '''
    根据不同动量系数和学习率绘制迭代示意图
    :param lr:
    :param gamma:
    :return:
    '''
    x0 = np.array([0., 1.])
    opts = optimizer(lr=lr, gamma=gamma)
    final_x, epoches, records_f, records_g, records_x = opts(x0)  # 调用optimizer中的__call__方法
    print('第四题最优点和最优值分别为：', final_x, records_f[:, -1])
    # 绘制等高线和更新点
    def height(x, y):
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
        # return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
    x = np.linspace(-6, 10, 300)
    y = np.linspace(-6, 10, 300)
    X, Y = np.meshgrid(x, y)
    # 为等高线填充颜色 10表示按照高度分成20层
    plt.contourf(X, Y, height(X, Y), 20, alpha=0.75, cmap=plt.cm.hot)
    x1 = records_x[0, :]
    x2 = records_x[1, :]
    plt.plot(x1[0], x2[0], '*')
    plt.plot(x1[1:-2], x2[1:-2], 'o')
    plt.plot(x1[-1], x2[-1], 's')
    plt.plot(x1, x2, linewidth=1, color='red')
    plt.title('learning rate={},gamma={},x1:{:.6f},x2:{:.6f}'.format(lr, gamma, final_x[0], final_x[1]), fontsize=10)
    plt.show()

if __name__ == '__main__':
    muti_sovler(lr=0.1,gamma=0.9)
    muti_sovler(lr=0.1, gamma=0.4)
    muti_sovler(lr=0.001, gamma=0.9)
    muti_sovler(lr=0.0001, gamma=0.9)







