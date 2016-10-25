"""
GPoFM: Gaussian Process Training with
       Optimized Feature Maps for Shift-Invariant Kernels
Github: https://github.com/MaxInGaussian/GPoFM
Author: Max W. Y. Lam [maxingaussian@gmail.com]
"""

import os, sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from GPoFM import *

BEST_MODEL_PATH = 'boston.pkl'

############################ Prior Setting ############################
reps_per_feats = 50
plot_metric = 'mse'
select_params_metric = 'cost'
select_model_metric = 'mse'
visualizer = None
# fig = plt.figure(figsize=(8, 6), facecolor='white')
# visualizer = Visualizer(fig, plot_metric)
nfeats_range = [10, 90]
algo = {
    'algo': 'adam',
    'algo_params': {
        'learning_rate':0.01,
        'beta1':0.9,
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
    'max_iter': 200
}

############################ General Methods ############################
def plot_dist(*args):
    import seaborn as sns
    for x in args:
        plt.figure()
        sns.distplot(x)
    plt.show()

def load_boston_data(prop=400/506):
    from sklearn import datasets
    boston = datasets.load_boston()
    X, y = boston.data, boston.target
    y = y[:, None]
    ntrain = y.shape[0]
    train_inds = npr.choice(range(ntrain), int(prop*ntrain), replace=False)
    valid_inds = np.setdiff1d(range(ntrain), train_inds)
    X_train, y_train = X[train_inds].copy(), y[train_inds].copy()
    X_valid, y_valid = X[valid_inds].copy(), y[valid_inds].copy()
    return X_train, y_train, X_valid, y_valid

############################ Training Phase ############################
X_train, y_train, X_valid, y_valid = load_boston_data()
nfeats_range_length = nfeats_range[1]-nfeats_range[0]
nfeats_choices = [nfeats_range[0]+(i*nfeats_range_length)//8 for i in range(5)]
evals = {
    'score': ['Model Selection Score', []],
    'obj': ['Params Optimization Objective', []],
    'mae': ['Mean Absolute Error', []],
    'nmae': ['Normalized Mean Absolute Error', []],
    'mse': ['Mean Square Error', []],
    'nmse': ['Normalized Mean Square Error', []],
    'mnlp': ['Mean Negative Log Probability', []],
    'time': ['Training Time(s)', []],
}
for nfeats in nfeats_choices:
    funcs = None
    results = {en:[] for en in evals.keys()}
    for round in range(reps_per_feats):
        X_train, y_train, X_valid, y_valid = load_boston_data()
        model = GPoFM(SCFGP(nfeats=nfeats))
        if(funcs is None):
            model.set_data(X_train, y_train)
            model.optimize(X_valid, y_valid, None, visualizer, **opt_params)
            funcs = model.get_compiled_funcs()
        else:
            model.set_data(X_train, y_train)
            model.optimize(X_valid, y_valid, funcs, visualizer, **opt_params)
        if(not os.path.exists(BEST_MODEL_PATH)):
            model.save(BEST_MODEL_PATH)
        else:
            best_model = GPoFM(Model().load(BEST_MODEL_PATH))
            best_model.predict(X_valid, y_valid)
            best_model._print_evals_comparison(model.evals)
            if(model.evals[select_model_metric][1][-1] <
                best_model.evals[select_model_metric][1][-1]):
                model.save(BEST_MODEL_PATH)
                print("!"*80)
                print("!"*30, "NEW BEST PREDICTOR", "!"*30)
                print("!"*80)
        for res in evals.keys():
            results[res].append(model.evals[res][1][-1])
    for en in evals.keys():
        evals[en][1].append((np.mean(results[en]), np.std(results[en])))

############################ Plot Performances ############################
import os
if not os.path.exists('plots'):
    os.mkdir('plots')
for en, (metric_name, metric_result) in evals.items():
    f = plt.figure(figsize=(8, 6), facecolor='white', dpi=120)
    ax = f.add_subplot(111)
    maxv, minv = 0, 1e5
    for i in range(len(nfeats_choices)):
        maxv = max(maxv, metric_result[i][0])
        minv = min(minv, metric_result[i][0])
        ax.text(nfeats_choices[i], metric_result[i][0], '%.2f' % (
            metric_result[i][0]), fontsize=5)
    line = ax.errorbar(nfeats_choices, [metric_result[i][0] for i in
        range(len(nfeats_choices))], fmt='-o')
    ax.set_xlim([min(nfeats_choices)-10, max(nfeats_choices)+10])
    ax.set_ylim([minv-(maxv-minv)*0.15,maxv+(maxv-minv)*0.45])
    plt.title(metric_name, fontsize=18)
    plt.xlabel('Number of Fourier Features', fontsize=13)
    plt.ylabel(en, fontsize=13)
    plt.savefig('plots/'+en.lower()+'.png')