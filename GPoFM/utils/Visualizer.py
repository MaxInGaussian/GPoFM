"""
GPoFM: Gaussian Process Training with
       Optimized Feature Maps for Shift-Invariant Kernels
Github: https://github.com/MaxInGaussian/GPoFM
Author: Max W. Y. Lam [maxingaussian@gmail.com]
"""

import numpy as np
import matplotlib.pyplot as plt

from theano import shared as Ts, function as Tf, tensor as TT
from theano.sandbox import linalg as Tlin

from .. import __init__

__all__ = [
    "Visualizer",
]

class Visualizer(object):
    
    " Visualizer (Data Visualization) "
    
    model, fig = None, None
    
    def __init__(self, fig, eval='nmse', plot_limit=80):
        self.fig = fig
        self.eval = eval.lower()
        self.plot_limit = plot_limit

    def similarity_plot(self):
        from sklearn.decomposition import PCA
        self.fig.suptitle(self.model.__str__(), fontsize=15)
        ax = self.fig.add_subplot(111)
        def animate(i):
            ax.cla()
            pts = 100
            Xs = Ts(np.tile(np.linspace(0., 1., pts), (self.model.D, 1)).T)
            Phi = self.model.feature_maps(Xs, self.model.params)[-1].eval()
            pca = PCA(n_components=2)
            X_r = pca.fit(Phi).transform(Phi)
            ax.contourf(X_r[:, 0], X_r[:, 1], self.model.yt.ravel())
            # surf = ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap=cm.jet,
            #                     linewidth=1, antialiased=True)
            # 
            # ax.set_zlim3d(np.min(Z), np.max(Z))
            # fig.colorbar(surf)
            ax.colorbar()
        return animate
            
    def train_plot(self):
        if(self.model.D == 1):
            return self.train_1d_plot()
        else:
            return self.train_eval_plot()
    
    def train_1d_plot(self):
        self.fig.suptitle(self.model.__str__(), fontsize=15)
        ax = self.fig.add_subplot(111)
        def animate(i):
            ax.cla()
            pts = 300
            errors = [0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64, 2.2]
            Xs = np.linspace(0., 1., pts)[:, None]
            pred_inputs = self.model.pack_train_func_inputs(Xs)
            predicted_mats = self.model.compiled_funcs['pred'](*pred_inputs)
            predicted_mats = self.model.unpack_predicted_mats(predicted_mats)
            mu, std = predicted_mats['mu_f'], predicted_mats['std_f']
            for er in errors:
                ax.fill_between(Xs[:, 0], mu-er*std, mu+er*std,
                                alpha=((3-er)/5.5)**1.7, facecolor='blue',
                                linewidth=1e-3)
            ax.plot(Xs[:, 0], mu, alpha=0.8, c='black')
            ax.errorbar(self.model.Xt[:, 0],
                self.model.yt.ravel(), fmt='r.', markersize=5, alpha=0.6)
            yrng = self.model.yt.max()-self.model.yt.min()
            ax.set_ylim([self.model.yt.min(), self.model.yt.max()+0.5*yrng])
            ax.set_xlim([-0.1, 1.1])
        return animate
    
    def train_eval_plot(self):
        self.fig.suptitle(self.model.__str__(), fontsize=15)
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        plt.xlabel('TIME(s)', fontsize=13)
        def animate(i):
            if(i == 0):
                data_x1, data_y1, data_x2, data_y2 = [], [], [], []
            else:
                data_x1 = ax1.lines[0].get_xdata().tolist()
                data_y1 = ax1.lines[0].get_ydata().tolist()
                data_x2 = ax2.lines[0].get_xdata().tolist()
                data_y2 = ax2.lines[0].get_ydata().tolist()
            data_x1.append(self.model.evals['time'][1][-1])
            obj = self.model.evals['obj'][1][self.model.evals_ind]
            data_y1.append(obj)
            ax1.cla()
            ax1.plot(data_x1[-self.plot_limit:], data_y1[-self.plot_limit:],
                color='r', linewidth=2.0, label='MIN OBJ')
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels, loc='upper center',
                bbox_to_anchor=(0.5, 1.05), ncol=1, fancybox=True)   
            data_x2.append(self.model.evals['time'][1][-1])
            val = self.model.evals[self.eval][1][self.model.evals_ind]
            data_y2.append(val)          
            ax2.cla()
            ax2.plot(data_x2[-self.plot_limit:], data_y2[-self.plot_limit:],
                color='b', linewidth=2.0, label=self.eval.upper())
            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(handles, labels, loc='upper center',
                bbox_to_anchor=(0.5, 1.05), ncol=1, fancybox=True)
        return animate