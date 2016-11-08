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
from theano import tensor as TT

from . import Model

__all__ = [
    "GPoHF",
    "GPoTHF",
]

class GPoHF(Model):
    
    '''
    The :class:`GPoHF` class implemented a GPoFM model:
        Gaussian process Optimizing Homogeneous Feature Maps (GPoHF)
    
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

    def randomized_params(self):
        S = self.setting['nfeats']
        const = npr.randn(2)*1e-2
        l = npr.randn(self.D)
        g = npr.randn(self.D*S)
        f = npr.randn(self.D*S)
        p = 2*np.pi*npr.rand(S)
        return [const, l, g, f, p]
    
    def feature_maps(self, X, params):
        t_ind, S = 0, self.setting['nfeats']
        a = params[0]; t_ind+=1; b = params[1]; t_ind+=1
        sig2_n, sig2_f = TT.exp(2*a), TT.exp(b)
        l = params[t_ind:t_ind+self.D]; t_ind+=self.D
        g = params[t_ind:t_ind+self.D*S]; t_ind+=self.D*S
        G = TT.reshape(g, (self.D, S))/np.exp(l[:, None])
        f = params[t_ind:t_ind+self.D*S]; t_ind+=self.D*S
        F = TT.reshape(f, (self.D, S))/np.exp(l[:, None])
        p = params[t_ind:t_ind+S]; t_ind+=S
        P = TT.reshape(p, (1, S))-TT.mean(F, 0)[None, :]
        FF = TT.dot(X, F)+P
        Phi = TT.cos(FF)*TT.exp(X.dot(G))*TT.sqrt(sig2_f/FF.shape[1])
        if(type(X) == TT.TensorVariable):
            return sig2_n, sig2_f, FF, Phi
        return Phi

class GPoTHF(GPoHF):
    
    '''
    The :class:`GPoTHF` class implemented a GPoFM model:
        Gaussian process Optimizing Transformed Fourier Feature Maps (GPoTHF)
    
    Parameters
    ----------
    nfeats : an integer
        Number of Fourier Features (nfeats > ncorr)
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
    
    def __init__(self, **args):
        super(GPoTHF, self).__init__(**args)
    
    def __str__(self):
        return "GPoTHF (Fourier = %d)"%(self.setting['nfeats'])

    def transform_inputs(self, params):
        sign = lambda x: TT.tanh(x*1e3)
        cdf = lambda x: .5*(1+T.erf(x/T.sqrt(2+epsilon)+epsilon))
        X = TT.dmatrices('X')
        X_lm = params[-(self.D+1):-1][None, :]
        X = (sign(X)*TT.sqrt(X**2)**X_lm-1)/X_lm
        return cdf(X)

    def transform_outputs(self, params, inverse=None):
        sign = lambda x: TT.tanh(x*1e3)
        y_lm = params[-1]
        if(inverse is not None):
            ty = inverse*y_lm+1
            ty = sign(ty)*TT.sqrt(ty**2)**(1./y_lm)
            return ty
        y = TT.dmatrices('y')
        ty = (sign(y)*TT.sqrt(y**2)**y_lm-1)/y_lm
        return y

    def randomized_params(self):
        lm = 2*np.pi*npr.rand(self.D+1)
        return super(GPoTHF, self).randomized_params()+[lm]