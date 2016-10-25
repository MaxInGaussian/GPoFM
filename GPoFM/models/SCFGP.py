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
    "SCFGP",
]

class SCFGP(Model):
    
    '''
    The :class:`SCFGP` class implemented handy functions shared by all machine
    learning models. It is always called as a subclass for any new model.
    
    Parameters
    ----------
    sparsity : an integer
        Sparsity of frequency matrix
    nfeats : an integer
        Number of correlated Fourier features
    X_trans : a string
        Transformation method used for inputs of training data
    y_trans : a string
        Transformation method used for outpus of training data
    verbose : a bool
        Idicator that determines whether printing training message or not
    '''
    
    setting, compiled_funcs = None, None
    
    def __init__(self, sparsity=20, nfeats=50, **args):
        super(SCFGP, self).__init__(**args)
        self.setting['sparsity'] = sparsity
        self.setting['nfeats'] = nfeats
    
    def __str__(self):
        S, M = self.setting['sparsity'], self.setting['nfeats']
        return "SCFGP (Sparsity=%d, Fourier Features=%d)"%(S, M)

    def init_params(self):
        S, M = self.setting['sparsity'], self.setting['nfeats']
        const = npr.rand(3)+1e-2
        l_f = npr.randn(self.D*S)
        r_f = npr.rand(M*S)
        l_p = 2*np.pi*npr.rand(S)
        p = 2*np.pi*npr.rand(M)
        self.params = Ts(np.concatenate([const, l_f, r_f, l_p, p]))
    
    def unpack_params(self, params):
        S, M = self.setting['sparsity'], self.setting['nfeats']
        t_ind = 0
        a = params[0];t_ind+=1
        b = params[1];t_ind+=1
        c = params[2];t_ind+=1
        l_f = params[t_ind:t_ind+self.D*S];t_ind+=self.D*S
        l_F = TT.reshape(l_f, (self.D, S))
        r_f = params[t_ind:t_ind+M*S];t_ind+=M*S
        r_F = TT.reshape(r_f, (M, S))/S
        F = l_F.dot(r_F.T)
        l_p = params[t_ind:t_ind+S];t_ind+=S
        l_P = TT.reshape(l_p, (1, S))-TT.mean(l_F, 0)[None, :]
        p = params[t_ind:t_ind+M];t_ind+=M
        P = TT.reshape(p, (1, M))-TT.mean(F, 0)[None, :]
        return a, b, c, l_F, F, l_P, P
    
    def unpack_trained_mats(self, trained_mats):
        return {'obj': np.double(trained_mats[0]),
                'alpha': trained_mats[1],
                'Li': trained_mats[2],}
    
    def unpack_predicted_mats(self, predicted_mats):
        return {'mu_f': predicted_mats[0],
                'std_f': predicted_mats[1],}
    
    def pack_train_func_inputs(self, X, y):
        return [X, y]
    
    def pack_pred_func_inputs(self, Xs):
        return [Xs, self.trained_mats['alpha'], self.trained_mats['Li']]
    
    def compile_theano_funcs(self, opt_algo, opt_params):
        self.compiled_funcs = {}
        S, M = self.setting['sparsity'], self.setting['nfeats']
        epsilon = 1e-6
        kl = lambda mu, sig: sig+mu**2-TT.log(sig)
        X, y = TT.dmatrices('X', 'y')
        params = TT.dvector('params')
        a, b, c, l_F, F, l_P, P = self.unpack_params(params)
        sig2_n, sig_f = TT.exp(2*a), TT.exp(b)
        l_FF = TT.dot(X, l_F)+l_P
        FF = TT.concatenate((l_FF, TT.dot(X, F)+P), 1)
        Phi = TT.concatenate((TT.cos(FF), TT.sin(FF)), 1)
        Phi = sig_f*TT.sqrt(2./M)*Phi
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
            (y**2).sum()-(beta**2).sum())+2*(X.shape[0]-M)*a
        penelty = (kl(mu_w, sig_w)*M+kl(mu_l, sig_l)*S)/(S+M)
        obj = (nlml+penelty)/X.shape[0]
        grads = TT.grad(obj, params)
        updates = getattr(Optimizer, opt_algo)(self.params, grads, **opt_params)
        updates = getattr(Optimizer, 'apply_momentum')(updates, momentum=0.9)
        train_inputs = [X, y]
        train_outputs = [obj, alpha, Li]
        self.compiled_funcs['train'] = Tf(train_inputs, train_outputs,
            givens=[(params, self.params)])
        self.compiled_funcs['opt'] = Tf(train_inputs, train_outputs,
            givens=[(params, self.params)], updates=updates)
        Xs, Li, alpha = TT.dmatrices('Xs', 'Li', 'alpha')
        l_FFs = TT.dot(Xs, l_F)+l_P
        FFs = TT.concatenate((l_FFs, TT.dot(Xs, F)+P), 1)
        Phis = TT.concatenate((TT.cos(FFs), TT.sin(FFs)), 1)
        Phis = sig_f*TT.sqrt(2./M)*Phis
        mu_pred = TT.dot(Phis, alpha)
        std_pred = (noise*(1+(TT.dot(Phis, Li.T)**2).sum(1)))**0.5
        pred_inputs = [Xs, alpha, Li]
        pred_outputs = [mu_pred, std_pred]
        self.compiled_funcs['pred'] = Tf(pred_inputs, pred_outputs,
            givens=[(params, self.params)])
