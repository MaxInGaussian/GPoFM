import sys, os, string, time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from theano import  shared as Ts, function as Tf, tensor as TT
from theano.sandbox import linalg as Tlin

from .. import __init__

__all__ = [
    "SSGP",
]


class SSGP(Model):
    
    '''
    The :class:`SSGP` class implemented handy functions shared by all machine
    learning models. It is always called as a subclass for any new model.
    
    Parameters
    ----------
    X_scaling : a string
        The pre-scaling method used for inputs of training data
    y_scaling : a string
        The pre-scaling method used for outpus of training data
    verbose : a bool
        An idicator that determines whether printing training message or not
    '''
    
    def __init__(self, nfeats=50, **args):
        super(SSGP, self).__init__(**args)
        self.setting['nfeats'] = nfeats
    
    def __str__(self):
        return "SSGP (Fourier Features=d)"%(m)

    def init_params(self):
        import numpy.random as npr
        const = npr.rand(3)+1e-2
        l = npr.randn(self.D)
        f = npr.randn(self.D*m)
        p = 2*np.pi*npr.rand(m)
        self.params = Ts(np.concatenate([const, l, f, p]))
    
    def unpack_params(self, params):
        t_ind = 0
        a = hyper[0];t_ind+=1
        b = hyper[1];t_ind+=1
        c = hyper[2];t_ind+=1
        l = hyper[t_ind:t_ind+self.D];t_ind+=self.D
        f = hyper[t_ind:t_ind+self.D*m];t_ind+=self.D*m
        F = TT.reshape(f, (self.D, m))/np.exp(l[:, None])
        p = hyper[t_ind:t_ind+m];t_ind+=m
        P = TT.reshape(p, (1, m))-TT.mean(F, 0)[None, :]
        return a, b, c, F, P
    
    def unpack_trained_mats(self, trained_mats):
        return {'obj': trained_mats[0],
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
        eps, m = 1e-6, self.setting['nfeats']
        kl = lambda mu, sig: sig+mu**2-TT.log(sig)
        self.compiled_funcs = {}
        X, y = TT.dmatrices('X', 'y')
        params = TT.dvector('params')
        a, b, c, F, P = self.unpack_params(params)
        sig2_n, sig_f = TT.exp(2*a), TT.exp(b)
        FF = TT.dot(X, F)+P
        Phi = TT.concatenate((TT.cos(FF), TT.sin(FF)), 1)
        Phi = sig_f*TT.sqrt(2./m)*Phi
        noise = TT.log(1+TT.exp(c))
        PhiTPhi = TT.dot(Phi.T, Phi)
        A = PhiTPhi+(sig2_n+eps)*TT.identity_like(PhiTPhi)
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
            (y**2).sum()-(beta**2).sum())+2*(X.shape[0]-m)*a
        penelty = kl(mu_w, sig_w)
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
        FFs = TT.dot(Xs, F)+P)
        Phis = TT.concatenate((TT.cos(FFs), TT.sin(FFs)), 1)
        Phis = sig_f*TT.sqrt(2./m)*Phis
        mu_pred = TT.dot(Phis, alpha)
        std_pred = (noise*(1+(TT.dot(Phis, Li.T)**2).sum(1)))**0.5
        pred_inputs = [Xs, alpha, Li]
        pred_outputs = [mu_pred, std_pred]
        self.compiled_funcs['pred'] = Tf(pred_inputs, pred_outputs,
            givens=[(params, self.params)])
