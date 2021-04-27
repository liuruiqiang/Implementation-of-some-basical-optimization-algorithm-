import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

# def search_alpha(f,g,x,d,rph,init_alpha,coe):
#     '''
#     利用Amijio原则对alpha进行非精确线性搜索
#     '''
#     sigma = 0.4
#     gama = 0.5
#     k = 0
#     mk = 0

class optimizer(object):
    def __init__(self,f_mode='f',rph=0.1,init_alpha=1,t=2):
        self.f_mode = f_mode
        self.rph = rph
        self.init_alpha = init_alpha
        self.t = t

    def search_alpha(self,x,d,rph,init_alpha,coe):
        '''
        利用Goldstein原则对alpha进行非精确线性搜索
        '''
        flag = False
        a = 0
        b = init_alpha
        fx = self.f(x)
        gx = self.g(x)
        f0 = fx
        gd0 = np.dot(gx.T,d)
        alpha = b * np.random.uniform(0,1)
        k = 0
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
            k = k+1

        return alpha

    def f(self,x):
        '''
        返回第四题函数数值结果
        '''
        x1 ,x2 = x[0],x[1]
        return (x1**3 - x2)**2 + 2*(x2 - x1)**4

    def g(self,x):
        '''
        返回第四题函数梯度数值结果
        '''
        x1 ,x2 = x[0],x[1]
        g1 = 2*(x1**3 - x2)*3*x1**2 - 8*(x2-x1)**3
        g2 = 8 *(x2 - x1)**3 - 2*(x1**3 - x2)
        
        results = np.array([g1,g2])
        return results

    def BFGS_descend(self,x0,epoches=1000,eps=1e-6):
        '''
        针对第四题利用BFGS算法进行优化的过程
        '''
        fx0 = self.f(x0)
        grad0 = self.g(x0)
        lens = len(x0)
        H0 = np.identity(lens)#得到单位阵
        delta = np.sum(grad0**2)
        i = 1
        x = x0
       record_f = np.zeros((1,epoches))#存储迭代过程中的函数值
        record_g = np.zeros((lens,epoches))#存储迭代过程中的梯度值
        record_f[0,0] = fx0
        record_g[:,0] = grad0.flatten()
        while(i < epoches and delta >eps):
            d = -np.dot(H0,g(x0))
            gx = self.g(x0)
            
            alpha = self.search_alpha(x,d,self.rph,self.init_alpha,self.t)
            x = x0 + alpha * d
            gk = self.g(x)
            s = np.expand_dims((x - x0),1)
            y = np.expand_dims((gk - gx),1)
            x0 = x

            sy1 = np.dot(s,y.T)
            sy2 = np.dot(s.T,y)
            ss = np.dot(s,s.T)
            ones1 = np.identity(sy1.shape[0])#得到单位阵
            H1 = np.dot((ones1 - sy1/sy2),H0)
            H1 = np.dot(H1,(ones1 - sy1/sy2).T)
            H1 = H1 + ss/sy2
            H0 = H1#更新H
            fx = self.f(x)
            record_f[0,i] = fx
            record_g[:,i] = gk.flatten()
            delta = np.sum(gk **2)
            i += 1

        return x,i,record_f[:,:i],record_g[:,:i]

    def __call__(self,x0):
        final_x,epoches,records_f,records_g = self.BFGS_descend(x0)
        return final_x,epoches,records_f,records_g

if __name__ == '__main__':
    x0 = np.array([2.,3.])
    opts = optimizer()
    final_x,epoches,records_f,records_g = opts(x0)#调用optimizer中的__call__方法
    print('第四题最优点和最优值分别为：',final_x,records_f[:,-1])




