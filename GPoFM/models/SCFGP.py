################################################################################
#  GPoFM: Gaussian Process Training with
#  Optimized Feature Maps for Shift-Invariant Kernels
#  Github: https://github.com/MaxInGaussian/GPoFM
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import sys, os, string, time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from theano import  shared as Ts, function as Tf, tensor as TT
from theano.sandbox import linalg as Tlin

from .Scaler import Scaler
from .Optimizer import Optimizer as OPT

class SCFGP(object):
    
    """
    Sparsely Correlated Fourier Features Based Gaussian Process
    """

    ID, NAME, verbose = "", "", True
    X_scaler, y_scaler = [None]*2
    M, N, D = -1, -1, -1
    X, y, hyper, Li, alpha, train_func, pred_func = [None]*7
    
    
    def __init__(self, sparsity=20, nfeats=18, evals=None,
        X_scaling_method='auto-inv-normal',
        y_scaling_method='auto-normal', verbose=False):
        self.S = sparsity
        self.M = nfeats
        self.X_scaler = Scaler(X_scaling_method)
        self.y_scaler = Scaler(y_scaling_method)
        self.evals = {
            "SCORE": ["Model Selection Score", []],
            "COST": ["Hyperparameter Selection Cost", []],
            "MAE": ["Mean Absolute Error", []],
            "NMAE": ["Normalized Mean Absolute Error", []],
            "MSE": ["Mean Square Error", []],
            "NMSE": ["Normalized Mean Square Error", []],
            "MNLP": ["Mean Negative Log Probability", []],
            "TIME(s)": ["Training Time", []],
        } if evals is None else evals
        self.verbose = verbose
        self.generate_ID()
    
    def message(self, *arg):
        if(self.verbose):
            print(" ".join(map(str, arg)))
            sys.stdout.flush()
    
    def generate_ID(self):
        self.ID = ''.join(
            chr(npr.choice([ord(c) for c in (
                string.ascii_uppercase+string.digits)])) for _ in range(5))
        self.NAME = "SCFGP (Sparsity=%d, Fourier Features=%d)"%(self.S, self.M)

    def init_params(self):
        import numpy.random as npr
        const = npr.randn(3)
        l_f = npr.randn(self.D*self.S)
        r_f = npr.rand(self.M*self.S)
        l_p = 2*np.pi*npr.rand(self.S)
        p = 2*np.pi*npr.rand(self.M)
        self.params = Ts(np.concatenate([const, l_f, r_f, l_p, p]))
    
    def unpack_params(self, hyper):
        t_ind = 0
        a = hyper[0];t_ind+=1
        b = hyper[1];t_ind+=1
        c = hyper[2];t_ind+=1
        l_f = hyper[t_ind:t_ind+self.D*self.S];t_ind+=self.D*self.S
        l_F = TT.reshape(l_f, (self.D, self.S))
        r_f = hyper[t_ind:t_ind+self.M*self.S];t_ind+=self.M*self.S
        r_F = TT.reshape(r_f, (self.M, self.S))
        F = l_F.dot(r_F.T)
        l_p = hyper[t_ind:t_ind+self.S];t_ind+=self.S
        l_P = TT.reshape(l_p, (1, self.S))
        p = hyper[t_ind:t_ind+self.M];t_ind+=self.M
        P = TT.reshape(p, (1, self.M))
        l_FC = l_P-TT.mean(l_F, 0)[None, :]
        FC = P-TT.mean(F, 0)[None, :]
        return a, b, c, l_F, F, l_FC, FC
    
    def build_theano_models(self, algo, algo_params):
        epsilon = 1e-6
        kl = lambda mu, sig: sig+mu**2-TT.log(sig)
        X, y = TT.dmatrices('X', 'y')
        params = TT.dvector('params')
        a, b, c, l_F, F, l_FC, FC = self.unpack_params(params)
        sig2_n, sig_f = TT.exp(2*a), TT.exp(b)
        l_FF = TT.dot(X, l_F)+l_FC
        FF = TT.concatenate((l_FF, TT.dot(X, F)+FC), 1)
        Phi = TT.concatenate((TT.cos(FF), TT.sin(FF)), 1)
        Phi = sig_f*TT.sqrt(2./self.M)*Phi
        noise = TT.log(1+TT.exp(c))
        PhiTPhi = TT.dot(Phi.T, Phi)
        A = PhiTPhi+(sig2_n+epsilon)*TT.identity_like(PhiTPhi)
        L = Tlin.cholesky(A)
        Li = Tlin.matrix_inverse(L)
        PhiTy = Phi.T.dot(y)
        beta = TT.dot(Li, PhiTy)
        alpha = TT.dot(Li.T, beta)
        mu_f = TT.dot(Phi, alpha)
        var_f = (TT.dot(Phi, Li.T)**2).sum(1)[:, None]
        dsp = noise*(var_f+1)
        mu_l = TT.sum(TT.mean(l_F, axis=1))
        sig_l = TT.sum(TT.std(l_F, axis=1))
        mu_w = TT.sum(TT.mean(F, axis=1))
        sig_w = TT.sum(TT.std(F, axis=1))
        hermgauss = np.polynomial.hermite.hermgauss(30)
        herm_x = Ts(hermgauss[0])[None, None, :]
        herm_w = Ts(hermgauss[1]/np.sqrt(np.pi))[None, None, :]
        herm_f = TT.sqrt(2*var_f[:, :, None])*herm_x+mu_f[:, :, None]
        nlk = (0.5*herm_f**2.-y[:, :, None]*herm_f)/dsp[:, :, None]+0.5*(
            TT.log(2*np.pi*dsp[:, :, None])+y[:, :, None]**2/dsp[:, :, None])
        enll = herm_w*nlk
        nlml = 2*TT.log(TT.diagonal(L)).sum()+2*enll.sum()+1./sig2_n*(
            (y**2).sum()-(beta**2).sum())+2*(X.shape[0]-self.M)*a
        penelty = (kl(mu_w, sig_w)*self.M+kl(mu_l, sig_l)*self.S)/(self.S+self.M)
        cost = (nlml+penelty)/X.shape[0]
        grads = TT.grad(cost, params)
        updates = getattr(OPT, algo)(self.params, grads, **algo_params)
        updates = getattr(OPT, 'apply_nesterov_momentum')(updates, momentum=0.9)
        train_inputs = [X, y]
        train_outputs = [cost, alpha, Li]
        self.train_func = Tf(train_inputs, train_outputs,
            givens=[(params, self.params)])
        self.train_iter_func = Tf(train_inputs, train_outputs,
            givens=[(params, self.params)], updates=updates)
        Xs, Li, alpha = TT.dmatrices('Xs', 'Li', 'alpha')
        l_FFs = TT.dot(Xs, l_F)+l_FC
        FFs = TT.concatenate((l_FFs, TT.dot(Xs, F)+FC), 1)
        Phis = TT.concatenate((TT.cos(FFs), TT.sin(FFs)), 1)
        Phis = sig_f*TT.sqrt(2./self.M)*Phis
        mu_pred = TT.dot(Phis, alpha)
        std_pred = (noise*(1+(TT.dot(Phis, Li.T)**2).sum(1)))**0.5
        pred_inputs = [Xs, alpha, Li]
        pred_outputs = [mu_pred, std_pred]
        self.pred_func = Tf(pred_inputs, pred_outputs,
            givens=[(params, self.params)])
    
    def get_compiled_funcs(self):
        return self.train_func, self.train_iter_func, self.pred_func

    def set_data(self, X, y):
        """
        X: Normally Distributed Inputs
        Y: Normally Distributed Outputs
        """
        self.message("-"*60, "\nNormalizing SCFGP training data...")
        self.X_scaler.fit(X)
        self.y_scaler.fit(y)
        self.X = self.X_scaler.forward_transform(X)
        self.y = self.y_scaler.forward_transform(y)
        self.message("done.")
        self.N, self.D = self.X.shape
        if('train_func' not in self.__dict__.keys()):
            self.message("-"*60, "\nInitializing SCFGP hyperparameters...")
            self.init_params()
            self.message("done.")
        else:
            cost, self.alpha, self.Li = self.train_func(self.X, self.y)
    
    def minibatches(self, X, y, batchsize, shuffle=True):
        assert len(X) == len(y)
        if(shuffle):
            inds = np.arange(len(X))
            np.random.shuffle(inds)
        for start_ind in range(0, len(X)-batchsize+1, batchsize):
            if shuffle:
                batch = inds[start_ind:start_ind+batchsize]
            else:
                batch = slice(start_ind, start_ind+batchsize)
            yield X[batch], y[batch]

    def optimize(self, Xv=None, yv=None, funcs=None, visualizer=None, **args):
        obj = 'COST' if 'obj' not in args.keys() else args['obj'].upper()
        obj = 'COST' if obj not in self.evals.keys() else obj
        algo = {'algo': None} if 'algo' not in args.keys() else args['algo']
        nbatches = 1 if 'nbatches' not in args.keys() else args['nbatches']
        batchsize = 150 if 'batchsize' not in args.keys() else args['batchsize']
        cvrg_tol = 1e-4 if 'cvrg_tol' not in args.keys() else args['cvrg_tol']
        max_cvrg = 18 if 'max_cvrg' not in args.keys() else args['max_cvrg']
        max_iter = 500 if 'max_iter' not in args.keys() else args['max_iter']
        if(algo['algo'] not in OPT.algos):
            algo = {
                'algo': 'adam',
                'algo_params': {
                    'learning_rate':0.01,
                    'beta1':0.9,
                    'beta2':0.999,
                    'epsilon':1e-8
                }
            }
        for metric in self.evals.keys():
            self.evals[metric][1] = []
        if(funcs is None):
            self.message("-"*50, "\nCompiling SCFGP theano model...")
            self.build_theano_models(algo['algo'], algo['algo_params'])
            self.message("done.")
        else:
            self.train_func, self.train_iter_func, self.pred_func = funcs
        if(visualizer is not None):
            visualizer.model = self
            animate = visualizer.train_with_plot()
        if(Xv is None or yv is None):
            obj = 'COST'
            self.evals['MAE'][1].append(0)
            self.evals['NMAE'][1].append(0)
            self.evals['MSE'][1].append(0)
            self.evals['NMSE'][1].append(0)
            self.evals['MNLP'][1].append(0)
            self.evals['SCORE'][1].append(0)
        self.min_obj_ind = 0
        train_start_time = time.time()
        min_obj_val, argmin_params, cvrg_iter = np.Infinity, self.params, 0
        for iter in range(max_iter):
            if(nbatches > 1):
                cost_sum, params_list, batch_count = 0, [], 0
                for X, y in self.minibatches(self.X, self.y, batchsize):
                    params_list.append(self.params.get_value())
                    cost, self.alpha, self.Li = self.train_iter_func(X, y)
                    cost_sum += cost;batch_count += 1
                    if(batch_count == nbatches):
                        break
                self.params = Ts(np.median(np.array(params_list), axis=0))
                self.evals['COST'][1].append(np.double(cost_sum/batch_count))
            else:
                cost, self.alpha, self.Li = self.train_iter_func(self.X, self.y)
                self.evals['COST'][1].append(cost)
            self.evals['TIME(s)'][1].append(time.time()-train_start_time)
            if(Xv is not None and yv is not None):
                self.predict(Xv, yv)
            if(iter%(max_iter//10) == 1):
                self.message("-"*17, "VALIDATION ITERATION", iter, "-"*17)
                self._print_current_evals()
            if(visualizer is not None):
                animate(iter)
                plt.pause(0.05)
            obj_val = self.evals[obj][1][-1]
            if(obj_val < min_obj_val):
                if(min_obj_val-obj_val < cvrg_tol):
                    cvrg_iter += 1
                else:
                    cvrg_iter = 0
                min_obj_val = obj_val
                self.min_obj_ind = len(self.evals['COST'][1])-1
                argmin_params = self.params.copy()
            else:
                cvrg_iter += 1
            if(iter > 30 and cvrg_iter > max_cvrg):
                break
            elif(cvrg_iter > max_cvrg*0.5):
                randp = np.random.rand()*cvrg_iter/max_cvrg*0.5
                self.params = (1-randp)*self.params+randp*argmin_params
        self.params = argmin_params.copy()
        cost, self.alpha, self.Li = self.train_func(self.X, self.y)
        self.evals['COST'][1].append(np.double(cost))
        self.evals['TIME(s)'][1].append(time.time()-train_start_time)
        if(Xv is not None and yv is not None):
            self.predict(Xv, yv)
        self.min_obj_ind = len(self.evals['COST'][1])-1
        disp = self.verbose
        self.verbose = True
        self.message("-"*19, "OPTIMIZATION RESULT", "-"*20)
        self._print_current_evals()
        self.message("-"*60)
        self.verbose = disp

    def predict(self, Xs, ys=None):
        self.Xs = self.X_scaler.forward_transform(Xs)
        mu_f, std_f = self.pred_func(self.Xs, self.alpha, self.Li)
        mu_y = self.y_scaler.backward_transform(mu_f)
        up_bnd_y = self.y_scaler.backward_transform(mu_f+std_f[:, None])
        dn_bnd_y = self.y_scaler.backward_transform(mu_f-std_f[:, None])
        std_y = 0.5*(up_bnd_y-dn_bnd_y)
        if(ys is not None):
            self.evals['MAE'][1].append(np.mean(np.abs(mu_y-ys)))
            self.evals['NMAE'][1].append(self.evals['MAE'][1][-1]/np.std(ys))
            self.evals['MSE'][1].append(np.mean((mu_y-ys)**2.))
            self.evals['NMSE'][1].append(self.evals['MSE'][1][-1]/np.var(ys))
            self.evals['MNLP'][1].append(0.5*np.mean(((
                ys-mu_y)/std_y)**2+np.log(2*np.pi*std_y**2)))
            self.evals['SCORE'][1].append(
                self.evals['NMSE'][1][-1]/(1+np.exp(-self.evals['MNLP'][1][-1])))
        return mu_y, std_y

    def save(self, path):
        import pickle
        save_vars = ['ID', 'M', 'X_scaler', 'y_scaler', 'train_func',
            'pred_func', 'params', 'alpha', 'Li', 'evals']
        save_dict = {varn: self.__dict__[varn] for varn in save_vars}
        with open(path, "wb") as save_f:
            pickle.dump(save_dict, save_f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        import pickle
        with open(path, "rb") as load_f:
            load_dict = pickle.load(load_f)
        for varn, var in load_dict.items():
            self.__dict__[varn] = var
        self.NAME = "SCFGP (Sparsity=%d, Fourier Features=%d)"%(self.S, self.M)

    def _print_current_evals(self):
        for metric in sorted(self.evals.keys()):
            if(len(self.evals[metric][1]) < len(self.evals['COST'][1])):
                continue
            best_perform_eval = self.evals[metric][1][self.min_obj_ind]
            self.message(self.NAME, "%7s = %.4e"%(metric, best_perform_eval))




