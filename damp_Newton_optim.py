import numpy as np
import sympy as sym


class optimizer(object):
    def __init__(self,f_mode='f',rph=0.1,init_alpha=1,t=2):
        self.f_mode = f_mode
        self.rph = rph
        self.init_alpha = init_alpha
        self.t = t

    def Hessian(self,x,unknown=False):
        '''
        计算Hessian阵
        '''
        lens = len(x)
        s = 'x:'+str(lens+1)
        symp_x = sym.symbols(s)
        symp_x = symp_x[1:]
        match_dic = {}
        for i in range(lens):
            match_dic[symp_x[i]] = x[i]
        if unknown:
            fx,npfx = self.unknown_f(symp_x,x)
        else:
            fx = self.f(symp_x)
        H = sym.zeros(lens,lens)
        
        for j,r in enumerate(symp_x):
            for k , l in enumerate(symp_x):
                H[j,k] = sym.diff(sym.diff(fx,r),l).evalf(subs=match_dic)

        return np.array(H.tolist()).astype(np.float64)

    def get_symx(self,lens):
        result_x = 'x:'+str(lens+1)
        result_x = sym.symbols(result_x)
        return result_x[1:]

    def search_alpha(self,x,d,rph,init_alpha,coe):
        '''
        利用Goldstein原则对alpha进行非精确线性搜索
        '''
        flag = False
        a = 0
        b = init_alpha
        fx = f(x)
        gx = g(x)
        f0 = fx
        gd0 = np.dot(gx,d)
        # print("gx",gx,d,gd0)
        alpha = b * np.random.uniform(0,1)
        while(not flag):
            newfx = f(x + alpha *d)
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

    def f(self,x):
         '''
        返回第四题函数数值结果
        '''
        x1 ,x2 = x[0],x[1]
        # print('x1:',x1,x2)
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

    def damp_newton_opt(self,x0,epoches=1000,eps=1e-6):
        lens = len(x0)
        fx0 = self.f(x0)
        grad0 = self.g(x0)
        delta = np.sum(grad0**2)
        i = 1
        x = x0
        record_f = np.zeros((1,epoches))#存储迭代过程中的函数值
        record_g = np.zeros((lens,epoches))#存储迭代过程中的梯度值
        record_f[0,0] = fx0
        record_g[:,0] = grad0
        while(i < epoches and delta >eps):
            grad = self.g(x)
            Gx = self.Hessian(x,f)
            Gx_reverse = np.linalg.inv(Gx)
            dk = np.dot(-Gx_reverse,grad)
            alpha = self.search_alpha(x,dk,self.rph,self.init_alpha,self.t)
            x = x + alpha * dk
            record_f[0,i] = f(x)
            record_g[:,i] = g(x)
            delta = np.sum(grad**2)
            i = i + 1

        return x,i,record_f[:,:i],record_g[:,:i]

    def __call__(self,x0):
        final_x,epoches,records_f,records_g = self.damp_newton_opt(x0)
        return final_x,epoches,records_f,records_g


if __name__ == '__main__':
    x0 = np.array([2.,3.])
    opts = optimizer()
    final_x,epoches,records_f,records_g = opts(x0)#调用optimizer中的__call__方法
    print('第四题最优点和最优值分别为：',final_x,records_f[:,-1])


