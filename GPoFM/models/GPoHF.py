"""
GPoFM: Gaussian Process Training with
       Optimized Feature Maps for Shift-Invariant Kernels
Github: https://github.com/MaxInGaussian/GPoFM
Author: Max W. Y. Lam [maxingaussian@gmail.com]
"""

import sys, os, string, time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from theano import shared as Ts, function as Tf, tensor as TT
from theano.sandbox import linalg as Tlin

from . import Model
from .. import Optimizer

__all__ = [
    "GPoHF",
    "GPoCHF",
]


class GPoHF(Model):
    
    '''
    The :class:`GPoHF` class implemented a GPoFM model:
        Gaussian process Optimizing Homogeneous Features (GPoHF)
    
    Parameters
    ----------
    nfeats : an integer
        Number of Homogeneous Features
    penalty : a float
        Penalty for too complex function. default: 1.
    X_trans : a string
        Transformation method used for inputs of training data
    y_trans : a string
        Transformation method used for outpus of training data
    verbose : a bool
        Idicator that determines whether printing training message or not
    '''
    
    setting, compiled_funcs = None, None
    
    def __init__(self, nfeats=50, penalty=1., **args):
        super(GPoHF, self).__init__(**args)
        self.setting['nfeats'] = nfeats
        self.setting['penalty'] = penalty
    
    def __str__(self):
        return "GPoHF (Homogeneous = %d)"%(self.setting['nfeats'])

    def init_params(self):
        S = self.setting['nfeats']
        const = np.zeros(2)
        l = npr.randn(self.D)
        g = npr.randn(self.D*S)
        f = npr.randn(self.D*S)
        p = 2*np.pi*npr.rand(S)
        self.params = Ts(np.concatenate([const, l, g, f, p]))
    
    def unpack_params(self, params):
        t_ind, S = 0, self.setting['nfeats']
        a = params[0];t_ind+=1
        b = params[1];t_ind+=1
        l = params[t_ind:t_ind+self.D];t_ind+=self.D
        g = params[t_ind:t_ind+self.D*S];t_ind+=self.D*S
        G = TT.reshape(g, (self.D, S))/np.exp(l[:, None])
        f = params[t_ind:t_ind+self.D*S];t_ind+=self.D*S
        F = TT.reshape(f, (self.D, S))/np.exp(l[:, None])
        p = params[t_ind:t_ind+S];t_ind+=S
        P = TT.reshape(p, (1, S))-TT.mean(F, 0)[None, :]
        return a, b, G, F, P
    
    def unpack_trained_mats(self, trained_mats):
        return {'obj': np.double(trained_mats[0]),
                'alpha': trained_mats[1],
                'Li': trained_mats[2],
                'mu_f': trained_mats[3],}
    
    def unpack_predicted_mats(self, predicted_mats):
        return {'mu_fs': predicted_mats[0],
                'std_fs': predicted_mats[1],}
    
    def pack_train_func_inputs(self, X, y):
        return [X, y]
    
    def pack_pred_func_inputs(self, Xs):
        return [Xs, self.trained_mats['alpha'], self.trained_mats['Li']]
    
    def compile_theano_funcs(self, opt_algo, opt_params):
        self.compiled_funcs = {}
        eps, S = 1e-6, self.setting['nfeats']
        kl = lambda mu, std: TT.mean(std+mu**2-TT.log(std))
        X, y = TT.dmatrices('X', 'y')
        params = TT.dvector('params')
        a, b, G, F, P = self.unpack_params(params)
        sig2_n, sig_f = TT.exp(2*a), TT.exp(b)
        FF = TT.dot(X, F)+P
        Phi = TT.cos(FF)*TT.exp(X.dot(G))
        Phi = sig_f*TT.sqrt(1./S)*Phi
        PhiTPhi = TT.dot(Phi.T, Phi)
        A = PhiTPhi+(sig2_n+eps)*TT.identity_like(PhiTPhi)
        L = Tlin.cholesky(A)
        Li = Tlin.matrix_inverse(L)
        PhiTy = Phi.T.dot(y)
        beta = TT.dot(Li, PhiTy)
        alpha = TT.dot(Li.T, beta)
        mu_f = TT.dot(Phi, alpha)
        mu_w = TT.mean(F, axis=1)
        sig_w = TT.std(F, axis=1)
        nlml = 2*TT.log(TT.diagonal(L)).sum()+1./sig2_n*(
            (y**2).sum()-(beta**2).sum())+2*(X.shape[0]-S)*a
        penelty = kl(mu_w, sig_w)
        obj = (nlml+penelty*self.setting['penalty'])/X.shape[0]
        grads = TT.grad(obj, params)
        updates = getattr(Optimizer, opt_algo)(self.params, grads, **opt_params)
        updates = getattr(Optimizer, 'apply_momentum')(updates, momentum=0.9)
        train_inputs = [X, y]
        train_outputs = [obj, alpha, Li, mu_f]
        self.compiled_funcs['opt'] = Tf(train_inputs, train_outputs,
            givens=[(params, self.params)], updates=updates)
        self.compiled_funcs['train'] = Tf(train_inputs, train_outputs,
            givens=[(params, self.params)])
        Xs, Li, alpha = TT.dmatrices('Xs', 'Li', 'alpha')
        FFs = TT.dot(Xs, F)+P
        Phis = TT.cos(FFs)*TT.exp(Xs.dot(G))
        Phis = sig_f*TT.sqrt(1./S)*Phis
        mu_pred = TT.dot(Phis, alpha)
        std_pred = (sig2_n*(1+(TT.dot(Phis, Li.T)**2).sum(1)))**0.5
        pred_inputs = [Xs, alpha, Li]
        pred_outputs = [mu_pred, std_pred]
        self.compiled_funcs['pred'] = Tf(pred_inputs, pred_outputs,
            givens=[(params, self.params)])

class GPoCHF(GPoHF):
    
    '''
    The :class:`GPoCHF` class implemented a GPoFM model:
        Gaussian process Optimizing Correlated Homogeneous Features (GPoCHF)
    
    Parameters
    ----------
    ncorr : an integer
        Number of correlated Homogeneous features
    nfeats : an integer
        Number of Homogeneous Features (nfeats > ncorr)
    penalty : a float
        Penalty for too complex function. default: 1.
    X_trans : a string
        Transformation method used for inputs of training data
    y_trans : a string
        Transformation method used for outpus of training data
    verbose : a bool
        Idicator that determines whether printing training message or not
    '''
    
    setting, compiled_funcs = None, None
    
    def __init__(self, ncorr=10, **args):
        super(GPoCHF, self).__init__(**args)
        self.setting['ncorr'] = ncorr
    
    def __str__(self):
        S, M = self.setting['nfeats'], self.setting['ncorr']
        return "GPoCHF (Homogeneous = %d, Corr. Homogeneous = %d)"%(S, M)

    def init_params(self):
        S, M = self.setting['nfeats'], self.setting['ncorr']
        const = np.zeros(2)
        l = npr.randn(self.D)
        l_f = npr.randn(self.D*M)
        r_f = npr.rand(M*S)
        p = 2*np.pi*npr.rand(S)
        self.params = Ts(np.concatenate([const, l, l_f, r_f, p]))
    
    def unpack_params(self, params):
        S, M = self.setting['nfeats'], self.setting['ncorr']
        t_ind = 0
        a = params[0];t_ind+=1
        b = params[1];t_ind+=1
        l = params[t_ind:t_ind+self.D];t_ind+=self.D
        l_f = params[t_ind:t_ind+self.D*M];t_ind+=self.D*M
        l_F = TT.reshape(l_f, (self.D, M))
        r_f = params[t_ind:t_ind+M*S];t_ind+=M*S
        r_F = TT.reshape(r_f, (S, M))/M
        F = l_F.dot(r_F.T)/np.exp(l[:, None])
        p = params[t_ind:t_ind+S];t_ind+=S
        P = TT.reshape(p, (1, S))-TT.mean(F, 0)[None, :]
        return a, b, F, P
            
            
            