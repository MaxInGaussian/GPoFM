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

############################ Prior Setting ############################    
use_models = ['GPoFour']
resolution = 0.5
penalty = 0.5
dropout = 1.
transform = True
nfeats_choice = [10]
plot_metric = 'mse'
select_params_metric = 'nmse'
select_model_metric = 'score'
fig = plt.figure(figsize=(8, 6), facecolor='white')
visualizer = Visualizer(fig, plot_metric)
algo = {
    'algo': 'adam',
    'algo_params': {
        'learning_rate':0.01,
        'beta1':0.6,
        'beta2':0.19,
        'epsilon':1e-8
    }
}
opt_params = {
    'obj': select_params_metric,
    'algo': algo,
    'dropout': dropout,
    'cv_nfolds': 3,
    'cvrg_tol': 1e-6,
    'max_cvrg': 28,
    'max_iter': 1000
}
evals = {
    'score': [
        'Model Selection Score',
        {model_name: [] for model_name in use_models}
    ],
    'obj': [
        'Params Optimization Objective',
        {model_name: [] for model_name in use_models}
    ],
    'mae': [
        'Mean Absolute Error',
        {model_name: [] for model_name in use_models}
    ],
    'nmae': [
        'Normalized Mean Absolute Error',
        {model_name: [] for model_name in use_models}
    ],
    'mse': [
        'Mean Square Error',
        {model_name: [] for model_name in use_models}
    ],
    'nmse': [
        'Normalized Mean Square Error',
        {model_name: [] for model_name in use_models}
    ],
    'mnlp': [
        'Mean Negative Log Probability',
        {model_name: [] for model_name in use_models}
    ],
    'time': [
        'Training Time(s)',
        {model_name: [] for model_name in use_models}
    ],
}
        
############################ General Methods ############################
def plot_dist(*args):
    import seaborn as sns
    for x in args:
        plt.figure()
        sns.distplot(x)
    plt.show()
    
def load_co2_data(prop=1.):
    from sklearn.datasets import fetch_mldata
    from sklearn import cross_validation
    data = fetch_mldata('mauna-loa-atmospheric-co2').data
    X = data[:, [1]]
    y = data[:, 0]
    y = y[:, None]
    X = X.astype(np.float64)
    return X, y, X, y

############################ Training & Visualizing ############################
X_train, y_train, X_valid, y_valid = load_co2_data()
for nfeats in nfeats_choice:
    for model_name in use_models:
        ModelClass = getattr(sys.modules['GPoFM'], model_name)
        model = GPoFM(ModelClass(nfeats, resolution, penalty, transform))
        model.optimize(X_train, y_train, None, visualizer, **opt_params)
    
    
    
    
    
    
    
    
    
    
    
    
    