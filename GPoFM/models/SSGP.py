"""
GPoFS: Gaussian Process Training with
       Optimized Feature Saps for Shift-Invariant Kernels
Github: https://github.com/SaxInGaussian/GPoFS
Author: Sax W. Y. Lam [maxingaussian@gmail.com]
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
    "SSGP",
]


class SSGP(Model):
    
    '''
    The :class:`SSGP` class implemented handy functions shared by all machine
    learning models. It is always called as a subclass for any new model.
    
    Parameters
    ----------
    sparsity : an integer
        Sparsity of frequency matrix
    X_trans : a string
        Transformation method used for inputs of training data
    y_trans : a string
        Transformation method used for outpus of training data
    verbose : a bool
        Idicator that determines whether printing training message or not
    '''
    
    setting, compiled_funcs = None, None
    
    def __init__(self, sparsity=50, **args):
        super(SSGP, self).__init__(**args)
        self.setting['sparsity'] = sparsity
    
    def __str__(self):
        return "SSGP (Sparsity = %d)"%(self.setting['sparsity'])

    def init_params(self):
        S = self.setting['sparsity']
        const = np.zeros(2)
        l = npr.randn(self.D)
        f = npr.randn(self.D*S)
        p = 2*np.pi*npr.rand(S)
        self.params = Ts(np.concatenate([const, l, f, p]))
    
    def unpack_params(self, params):
        t_ind, S = 0, self.setting['sparsity']
        a = params[0];t_ind+=1
        b = params[1];t_ind+=1
        l = params[t_ind:t_ind+self.D];t_ind+=self.D
        f = params[t_ind:t_ind+self.D*S];t_ind+=self.D*S
        F = TT.reshape(f, (self.D, S))/np.exp(l[:, None])
        p = params[t_ind:t_ind+S];t_ind+=S
        P = TT.reshape(p, (1, S))-TT.mean(F, 0)[None, :]
        return a, b, F, P
    
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
        eps, S = 1e-6, self.setting['sparsity']
        kl = lambda mu, sig: sig+mu**2-TT.log(sig)
        X, y = TT.dmatrices('X', 'y')
        params = TT.dvector('params')
        a, b, F, P = self.unpack_params(params)
        sig2_n, sig_f = TT.exp(2*a), TT.exp(b)
        FF = TT.dot(X, F)+P
        Phi = TT.concatenate((TT.cos(FF), TT.sin(FF)), 1)
        Phi = sig_f*TT.sqrt(2./S)*Phi
        PhiTPhi = TT.dot(Phi.T, Phi)
        A = PhiTPhi+(sig2_n+eps)*TT.identity_like(PhiTPhi)
        L = Tlin.cholesky(A)
        Li = Tlin.matrix_inverse(L)
        PhiTy = Phi.T.dot(y)
        beta = TT.dot(Li, PhiTy)
        alpha = TT.dot(Li.T, beta)
        mu_f = TT.dot(Phi, alpha)
        nlml = 2*TT.log(TT.diagonal(L)).sum()+1./sig2_n*(
            (y**2).sum()-(beta**2).sum())+2*(X.shape[0]-S)*a
        obj = nlml/X.shape[0]
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
        Phis = TT.concatenate((TT.cos(FFs), TT.sin(FFs)), 1)
        Phis = sig_f*TT.sqrt(2./S)*Phis
        mu_pred = TT.dot(Phis, alpha)
        std_pred = (sig2_n*(1+(TT.dot(Phis, Li.T)**2).sum(1)))**0.5
        pred_inputs = [Xs, alpha, Li]
        pred_outputs = [mu_pred, std_pred]
        self.compiled_funcs['pred'] = Tf(pred_inputs, pred_outputs,
            givens=[(params, self.params)])
