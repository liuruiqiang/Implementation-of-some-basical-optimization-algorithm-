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

    def unknownf_search_alpha(self,symx,x,d,rph,init_alpha,coe):
        '''
        针对第五题这种函数变量不定的情况利用Goldstein原则对alpha进行非精确线性搜索
        '''
        flag = False
        a = 0
        b = init_alpha
        sym_fx,npfx = self.unknown_f(symx,x)
        gx = self.unknown_g(symx,x,sym_fx)
        # fx = f(x)
        # gx = g(x)
        f0 = npfx
        
        gd0 = np.dot(gx,d)
        alpha = b * np.random.uniform(0,1)
        while(not flag):
            # sym_fx,newfx = unknown_f
            sym_fx,newfx = self.unknown_f(symx,x + alpha *d)
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
            # print(alpha)

        return alpha

    def unknown_f(self,x,npx):
        '''
        根据符号变量元素和实数值给出最后的函数表达式和函数值
        :param x :sympy的符号变量元素
        :param npx :numpy类型的变量数值
        :return fx : 函数的符号表达式
        :return npfx : 函数的数值结果
        '''
        lens = len(x)
        results = []
        match_dic = {}
        for i in range(lens-1):
            results.append((1-x[i])**2 + 100*(x[i+1]-x[i]**2)**2)
            match_dic[x[i]] = npx[i]#产生后面要用的subs字典，将npx中的数值赋值给符号类型数据x
        match_dic[x[lens-1]] = npx[lens-1]
        fx = results[0]
        for i in range(1,lens-1):
            fx = fx + results[i]
        npfx = float(fx.evalf(subs=match_dic))#将数值与符号变量匹配输入函数中得到函数结果
        return fx,npfx

    def unknown_g(self,x,npx,fx):
        '''
        根据符号变量元素和实数值以及函数表达式给出最后的梯度数值结果
        :param x :sympy的符号变量元素
        :param npx :numpy类型的变量数值
        :param fx : 函数的符号表达式
        :return : 函数的梯度数值结果
        '''
        lens = len(x)
        grad_results = []
        match_dic = {}
        for i in range(lens):
            grad_results.append(sym.diff(fx,x[i]))
            match_dic[x[i]] = npx[i]#产生后面要用的subs字典，将npx中的数值赋值给符号类型数据x
        
        np_results = []
        for i in range(lens):
            np_results.append(float(grad_results[i].evalf(subs=match_dic)))
        
        return np.array(np_results)

    def get_symx(self,lens):
        '''
        根据函数变量个数给出符号元素
        '''
        result_x = 'x:'+str(lens+1)
        result_x = sym.symbols(result_x)
        return result_x[1:]

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
            d = -np.dot(H0,self.g(x0))
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

    def unknown_BFGS_descend(self,x0,epoches=1000,eps=1e-6):
        '''
        针对第四题利用BFGS算法进行优化的过程
        '''
        lens = len(x0)
        symx = self.get_symx(lens)#根据变量个数得到符号变量
        sym_fx,npfx = self.unknown_f(symx,x0)#得到函数符号表达式以及函数数值
        grad0  = self.unknown_g(symx,x0,sym_fx)#得到函数梯度数值
        delta = np.sum(grad0**2)
        H0 = np.identity(lens)#得到单位阵
        i = 1
        x = x0
        record_f = np.zeros((1,epoches))#存储迭代过程中的函数值
        record_g = np.zeros((lens,epoches))#存储迭代过程中的梯度值
        record_x = np.zeros((lens,epoches))#存储迭代过程中的更新变量
        record_normG = np.zeros((1,epoches))#存储梯度范数
        record_f[0,0] = npfx
        record_g[:,0] = grad0
        record_x[:,0] = x
        record_normG[0,0] = np.linalg.norm(grad0)
        while(i < epoches and delta >eps):
            gx = self.unknown_g(symx,x0,sym_fx)
            d = -np.dot(H0,gx)
            alpha = self.unknownf_search_alpha(symx,x,d,self.rph,self.init_alpha,self.t)
            # alpha = self.search_alpha(x,d,self.rph,self.init_alpha,self.t)
            x = x0 + alpha * d
            record_x[:,i] = x
            gk = self.unknown_g(symx,x,sym_fx)
            # gk = self.g(x)
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
            sym_fx,npfx = self.unknown_f(symx,x)
            fx = npfx
            record_f[0,i] = fx
            record_g[:,i] = gk.flatten()
            record_normG[0,i] = np.linalg.norm(gk)
            delta = np.sum(gk **2)
            i += 1

        return x,i,record_f[:,:i],record_g[:,:i],record_x[:,:i],record_normG[:,:i]

    def __call__(self,x0):
        if self.f_mode == 'f':
            final_x,epoches,records_f,records_g = self.BFGS_descend(x0)
            return final_x,epoches,records_f,records_g
        else:
            final_x,epoches,records_f,records_g,records_x,records_normG = self.unknown_BFGS_descend(x0)
            return final_x,epoches,records_f,records_g,records_x,records_normG


if __name__ == '__main__':
    x0 = np.array([2.,3.])
    opts = optimizer()
    final_x,epoches,records_f,records_g = opts(x0)#调用optimizer中的__call__方法
    print('第四题最优点和最优值分别为：',final_x,records_f[:,-1])
    lens = 2
    x0 = np.zeros((lens,))
    x0[0] = 1
    opts = optimizer(f_mode='unknown')
    final_x,epoches,records_f,records_g,records_x,record_normG = opts(x0)
    print('第五题N=2时最优点和最优值分别为：',final_x,records_f[:,-1])
    
#绘制等高线和更新点
    def height(x, y):
        return (1-x)**2 +100 *(y-x**2)**2
        # return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    # 为等高线填充颜色 10表示按照高度分成20层
    plt.contourf(X, Y, height(X, Y), 20, alpha=0.75, cmap=plt.cm.hot)
    x1 = records_x[0,:]
    x2 = records_x[1,:]
    plt.plot(x1[0],x2[0],'*')
    plt.plot(x1[1:-2],x2[1:-2],'o')
    plt.plot(x1[-1],x2[-1],'s')
    plt.plot(x1,x2,linewidth=1,color='red')
    plt.title('contour line',fontsize=30)
    plt.show()

    #N=7时求解最优解并显示函数值和梯度值
    lens = 7
    x0 = np.zeros((lens,))
    x0[0] = 1
    opts = optimizer(f_mode='unknown')
    final_x,epoches,records_f,records_g,records_x,record_normG = opts(x0)
    print('第五题N=7时最优点和最优值分别为：',final_x,records_f[:,-1])
    if epoches > 500:
        epoches = 500
    epos = np.linspace(0,epoches,epoches)
    plt.subplot(1,2,1)
    plt.plot(epos,records_f[0,:epoches],label='function values')
    plt.xlabel('epoch')
    plt.ylabel('function values')
    plt.text(-0.5,3,"function values",fontsize=20,color="red")
    plt.subplot(1,2,2)
    plt.plot(epos,record_normG[0,:epoches],label='function gradients')
    plt.xlabel('epoch')
    plt.ylabel('function gradients')
    plt.text(-0.5,3,"grad_descend",fontsize=20,color="red")
    plt.show()

    #N=20时求解最优解并显示函数值和梯度值
    lens = 20
    x0 = np.zeros((lens,))
    x0[0] = 1
    opts = optimizer(f_mode='unknown')
    final_x,epoches,records_f,records_g,records_x,record_normG = opts(x0)
    print('第五题N=20时最优点和最优值分别为：',final_x,records_f[:,-1])
    if epoches > 800:
        epoches = 800
    epos = np.linspace(0,epoches,epoches)
    plt.subplot(1,2,1)
    plt.plot(epos,records_f[0,:epoches],label='function values')
    plt.xlabel('epoch')
    plt.ylabel('function values')
    plt.text(-0.5,3,"function values",fontsize=20,color="red")
    plt.subplot(1,2,2)
    plt.plot(epos,record_normG[0,:epoches],label='function gradients')
    plt.xlabel('epoch')
    plt.ylabel('function gradients')
    plt.text(-0.5,3,"grad_descend",fontsize=20,color="red")
    plt.show()




