################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
from scipy.stats import skew, norm, mstats
from scipy.optimize import minimize

from .. import __init__

__all__ = [
    "Scaler",
]

class Scaler(object):
    
    " Scaler (Data Preprocessing) "

    algos = [
        "min-max",
        "normal",
        "inv-normal",
        "auto-normal",
        "auto-inv-normal",
    ]
    
    data = {}
    
    def __init__(self, algo):
        assert algo.lower() in self.algos, "Invalid Scaling Algorithm!"
        self.algo = algo.lower()
        if(self.algo == "min-max"):
            self.data = {"cols": None, "min": 0, "max":0}
        elif(self.algo == "normal"):
            self.data = {"cols": None, "std": 0, "mu":0}
        elif(self.algo == "inv-normal"):
            self.data = {"cols": None, "std": 0, "mu":0}
        elif(self.algo == "auto-normal"):
            self.data = {"cols": None, "min": 0, "max":0, "std": 0, "mu":0, "boxcox":0}
        elif(self.algo == "auto-inv-normal"):
            self.data = {"cols": None, "min": 0, "max":0, "std": 0, "mu":0, "boxcox":0}
    
    def fit(self, X):
        self.data["cols"] = list(set(range(X.shape[1])).difference(
            np.where(np.all(X == X[0,:], axis = 0))[0]))
        tX = X[:, self.data["cols"]]
        if(self.algo == "min-max"):
            self.data['min'] = np.min(tX, axis=0)
            self.data['max'] = np.max(tX, axis=0)
        elif(self.algo == "normal"):
            self.data['mu'] = np.mean(tX, axis=0)
            self.data['std'] = np.std(tX, axis=0)
        elif(self.algo == "inv-normal"):
            self.data['mu'] = np.mean(tX, axis=0)
            self.data['std'] = np.std(tX, axis=0)
        elif(self.algo == "auto-normal"):
            self.data['min'] = np.min(tX, axis=0)
            self.data['max'] = np.max(tX, axis=0)
            tX = (tX-self.data["min"])/(self.data["max"]-self.data["min"])
            boxcox = lambda x, lm: (np.sign(x)*np.abs(x)**lm-1)/lm
            self.data['boxcox'] = np.zeros(tX.shape[1])
            for d in range(tX.shape[1]):
                Xd = tX[:, d]
                if(np.unique(tX[:, d]).shape[0] < 10):
                    self.data['boxcox'][d] = 1
                    continue
                skewness = lambda x: skew(x, bias=False)**2
                t_lm = lambda lm: np.log(np.exp(lm[0])+1)
                boxcox_Xd = lambda lm: boxcox(Xd, t_lm(lm))
                obj = lambda lm: skewness(boxcox_Xd(lm))
                bounds = [(-5, 5)]
                lm = minimize(obj, [0.], method='SLSQP', bounds=bounds,
                    options={'ftol': 1e-8, 'maxiter':100, 'disp':False})['x']
                self.data['boxcox'][d] = t_lm(lm)
            lm = self.data['boxcox'][None, :]
            tX = boxcox(tX, lm)
            self.data['mu'] = np.mean(tX, axis=0)
            self.data['std'] = np.std(tX, axis=0)
        elif(self.algo == "auto-inv-normal"):
            self.data['min'] = np.min(tX, axis=0)
            self.data['max'] = np.max(tX, axis=0)
            tX = (tX-self.data["min"])/(self.data["max"]-self.data["min"])
            boxcox = lambda x, lm: (np.sign(x)*np.abs(x)**lm-1)/lm
            self.data['boxcox'] = np.zeros(tX.shape[1])
            for d in range(tX.shape[1]):
                Xd = tX[:, d]
                if(np.unique(tX[:, d]).shape[0] < 10):
                    self.data['boxcox'][d] = 1
                    continue
                skewness = lambda x: skew(x, bias=False)**2
                t_lm = lambda lm: np.log(np.exp(lm[0])+1)
                boxcox_Xd = lambda lm: boxcox(Xd, t_lm(lm))
                obj = lambda lm: skewness(boxcox_Xd(lm))
                bounds = [(-5, 5)]
                lm = minimize(obj, [0.], method='SLSQP', bounds=bounds,
                    options={'ftol': 1e-8, 'maxiter':100, 'disp':False})['x']
                self.data['boxcox'][d] = t_lm(lm)
            lm = self.data['boxcox'][None, :]
            tX = boxcox(tX, lm)
            self.data['mu'] = np.mean(tX, axis=0)
            self.data['std'] = np.std(tX, axis=0)
    
    def forward_transform(self, X):
        tX = X[:, self.data["cols"]]
        if(self.algo == "min-max"):
            return (tX-self.data["min"])/(self.data["max"]-self.data["min"])
        elif(self.algo == "normal"):
            return (tX-self.data["mu"])/self.data["std"]
        elif(self.algo == "inv-normal"):
            return norm.cdf((tX-self.data["mu"])/self.data["std"])
        elif(self.algo == "auto-normal"):
            tX = (tX-self.data["min"])/(self.data["max"]-self.data["min"])
            lm = self.data['boxcox'][None, :]
            boxcox = lambda x: (np.sign(x)*np.abs(x)**lm-1)/lm
            return (boxcox(tX)-self.data["mu"])/self.data["std"]
        elif(self.algo == "auto-inv-normal"):
            tX = (tX-self.data["min"])/(self.data["max"]-self.data["min"])
            lm = self.data['boxcox'][None, :]
            boxcox = lambda x: (np.sign(x)*np.abs(x)**lm-1)/lm
            return norm.cdf(boxcox(tX), self.data["mu"], self.data["std"])
    
    def backward_transform(self, X):
        assert len(self.data["cols"]) == X.shape[1], "Backward Transform Error"
        if(self.algo == "min-max"):
            return X*(self.data["max"]-self.data["min"])+self.data["min"]
        elif(self.algo == "normal"):
            return X*self.data["std"]+self.data["mu"]
        elif(self.algo == "inv-normal"):
            return (norm.ppf(X)-self.data["mu"])/self.data["std"]
        elif(self.algo == "auto-normal"):
            lm = self.data['boxcox'][None, :]
            inv_boxcox = lambda x: np.sign(x*lm+1)*np.abs(x*lm+1)**(1./lm)
            tX = X*self.data["std"]+self.data["mu"]
            return inv_boxcox(tX)*(self.data["max"]-self.data["min"])+self.data["min"]
        elif(self.algo == "auto-inv-normal"):
            lm = self.data['boxcox'][None, :]
            inv_boxcox = lambda x: np.sign(x*lm+1)*np.abs(x*lm+1)**(1./lm)
            tX = norm.ppf(X, self.data["mu"], self.data["std"])
            return inv_boxcox(tX)*(self.data["max"]-self.data["min"])+self.data["min"]
    
