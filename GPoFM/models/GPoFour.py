"""
GPoFM: Gaussian Process Training with
       Optimized Feature Maps for Shift-Invariant Kernels
Github: https://github.com/MaxInGaussian/GPoFM
Author: Max W. Y. Lam [maxingaussian@gmail.com]
"""

import sys, os, string, time
import numpy as np
import numpy.random as npr
from theano import config as Tc
from theano import tensor as TT
from theano.sandbox import linalg as Tlin

from . import Model

__all__ = [
    "GPoFour",
]

class GPoFour(Model):
    
    '''
    The :class:`GPoFour` class implemented a GPoFM model:
        Gaussian process Optimizing Fourier Feature Maps (GPoFour)
    
    Parameters
    ----------
    nfeats : an integer
        Number of Fourier Features
    penalty : a float
        Penalty for prevention of overfitting
    transform : a bool
        Idicator that determines whether tranform the data before training
    X_trans : a string
        Transformation method used for inputs of training data
    y_trans : a string
        Transformation method used for outpus of training data
    verbose : a bool
        Idicator that determines whether printing training message or not
    '''
    
    setting, compiled_funcs = None, None
    
    def __init__(self, nfeats, resolution=0.5, penalty=1., transform=True, **args):
        super(GPoFour, self).__init__(nfeats, resolution, penalty, transform, **args)
    
    def __str__(self):
        return "GPoFour (%d, %.2f, %.2f)"%(self.setting['nfeats'], self.setting['resolution'], self.setting['penalty'])

    def randomized_params(self):
        S = self.setting['nfeats']
        rand_params = []
        const = npr.randn(2)*1e-2
        l = npr.randn(self.D)
        f = npr.randn(self.D*S)
        p = 2*np.pi*npr.rand(S)
        rand_params = [const, l, f, p]
        if(self.setting['transform']):
            lm = 2*np.pi*npr.rand(self.D+1)
            rand_params.append(lm)
        return rand_params
    
    def feature_maps(self, X, params):
        t_ind, S = 0, self.setting['nfeats']
        a = params[0]; t_ind+=1; b = params[1]; t_ind+=1
        sig2_n, sig2_f = 1/(1+TT.exp(a))*self.setting['resolution'], TT.exp(b)
        l = params[t_ind:t_ind+self.D]; t_ind+=self.D
        f = params[t_ind:t_ind+self.D*S]; t_ind+=self.D*S
        F = TT.reshape(f, (self.D, S))/np.exp(l[:, None])
        p = params[t_ind:t_ind+S]; t_ind+=S
        P = TT.reshape(p, (1, S))
        FF = TT.dot(X, F)+P
        Phi = TT.cos(FF)*TT.sqrt(sig2_f/FF.shape[1])
        if(type(X) == TT.TensorVariable):
            return sig2_n, sig2_f, FF, Phi
        return Phi
        
    def transform_inputs(self, params):
        if(not self.setting['transform']):
            return super(GPoFour, self).transform_inputs(params)
        sign = lambda x: TT.tanh(x*1e3)
        cdf = lambda x: .5*(1+T.erf(x/T.sqrt(2+epsilon)+epsilon))
        X = TT.dmatrices('X')
        X_lm = params[-(self.D+1):-1][None, :]
        X = (sign(X)*TT.sqrt(X**2)**X_lm-1)/X_lm
        return cdf(X)

    def transform_outputs(self, params, inverse=None):
        if(not self.setting['transform']):
            return super(GPoFour, self).transform_outputs(params)
        sign = lambda x: TT.tanh(x*1e3)
        y_lm = params[-1]
        if(inverse is not None):
            ty = inverse*y_lm+1
            ty = sign(ty)*TT.sqrt(ty**2)**(1./y_lm)
            return ty
        y = TT.dmatrices('y')
        ty = (sign(y)*TT.sqrt(y**2)**y_lm-1)/y_lm
        return y