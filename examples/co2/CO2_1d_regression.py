################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import os, sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from SCFGP import SCFGP, Visualizer

############################ Prior Setting ############################    
select_params_metric = 'cost'
fig = plt.figure(figsize=(8, 6), facecolor='white')
visualizer = Visualizer(fig)
nfeats_choices = [50]
sparsity_choices = [100]
algo = {
    'algo': 'adamax',
    'algo_params': {
        'learning_rate':0.05,
        'beta1':0.8,
        'beta2':0.999,
        'epsilon':1e-8
    }
}
opt_params = {
    'obj': select_params_metric,
    'algo': algo,
    'nbatches': 1,
    'cvrg_tol': 1e-5,
    'max_cvrg': 8,
    'max_iter': 1000
}

############################ General Methods ############################
def plot_dist(*args):
    import seaborn as sns
    for x in args:
        plt.figure()
        sns.distplot(x)
    plt.show()
    
def load_co2_data(prop=0.8):
    from sklearn.datasets import fetch_mldata
    from sklearn import cross_validation
    data = fetch_mldata('mauna-loa-atmospheric-co2').data
    X = data[:, [1]]
    y = data[:, 0]
    y = y[:, None]
    X = X.astype(np.float64)
    ntrain = y.shape[0]
    train_inds = npr.choice(range(ntrain), int(prop*ntrain), replace=False)
    valid_inds = np.setdiff1d(range(ntrain), train_inds)
    X_train, y_train = X[train_inds].copy(), y[train_inds].copy()
    X_valid, y_valid = X[valid_inds].copy(), y[valid_inds].copy()
    return X_train, y_train, X_valid, y_valid

############################ Training & Visualizing ############################
X_train, y_train, X_valid, y_valid = load_co2_data()
for sparsity in sparsity_choices:
    for nfeats in nfeats_choices:
        model = SCFGP(sparsity, nfeats)
        model.set_data(X_train, y_train)
        model.optimize(X_valid, y_valid, None, visualizer, **opt_params)