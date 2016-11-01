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
use_models = ['GPoFF', 'GPoTFF', 'GPoCFF',
              'GPoHF', 'GPoTHF', 'GPoCHF',
              'GPoAF', 'GPoTAF', 'GPoCAF']
num = 5
reps_per_nfeats = 10
penalty = 1
nfeats_range = [10, num*10]
nfeats_length = nfeats_range[1]-nfeats_range[0]
nfeats_choice = [nfeats_range[0]+(i*nfeats_length)//(num-1) for i in range(num)]
plot_metric = 'mse'
select_params_metric = 'score'
select_model_metric = 'score'
visualizer = None
# fig = plt.figure(figsize=(8, 6), facecolor='white')
# visualizer = Visualizer(fig, plot_metric)
algo = {
    'algo': 'adam',
    'algo_params': {
        'learning_rate':0.005,
        'beta1':0.9,
        'beta2':0.999,
        'epsilon':1e-8
    }
}
opt_params = {
    'obj': select_params_metric,
    'algo': algo,
    'cv_nfolds': 5,
    'cvrg_tol': 1e-5,
    'max_cvrg': 8,
    'max_iter': 200
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

def load_data(prop=400./506):
    from sklearn import datasets
    boston = datasets.load_boston()
    X, y = boston.data, boston.target
    y = y[:, None]
    ntrain = y.shape[0]
    train_inds = npr.choice(range(ntrain), int(prop*ntrain), replace=False)
    test_inds = np.setdiff1d(range(ntrain), train_inds)
    X_train, y_train = X[train_inds].copy(), y[train_inds].copy()
    X_test, y_test = X[test_inds].copy(), y[test_inds].copy()
    return X_train, y_train, X_test, y_test

############################ Training Phase ############################
X_train, y_train, X_test, y_test = load_data()
for i, nfeats in enumerate(nfeats_choice):
    for model_name in use_models:
        ModelClass = getattr(sys.modules['GPoFM'], model_name)
        funcs = None
        results = {en:[] for en in evals.keys()}
        for round in range(reps_per_nfeats):
            model = GPoFM(ModelClass(nfeats=nfeats, penalty=penalty))
            if(funcs is None):
                model.fit(X_train, y_train, None, visualizer, **opt_params)
                funcs = model.get_compiled_funcs()
            else:
                model.fit(X_train, y_train, funcs, visualizer, **opt_params)
            if(not os.path.exists(BEST_MODEL_PATH)):
                model.save(BEST_MODEL_PATH)
            else:
                best_model = GPoFM(Model().load(BEST_MODEL_PATH))
                best_model.set_training_data(X_train, y_train)
                best_model.evaluate(X_test, y_test)
                model.evaluate(X_test, y_test)
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
            eval = (np.mean(results[en]), np.std(results[en]))
            evals[en][1][model_name].append(eval)


############################ Plot Performances ############################
    import os
    if not os.path.exists('plots'):
        os.mkdir('plots')
    for en, (metric_name, metric_result) in evals.items():
        f = plt.figure(figsize=(8, 6), facecolor='white', dpi=120)
        ax = f.add_subplot(111)
        maxv, minv = 0, 1e5
        for model_name in metric_result.keys():
            for j in range(i+1):
                maxv = max(maxv, metric_result[model_name][j][0])
                minv = min(minv, metric_result[model_name][j][0])
                ax.text(nfeats_choice[j], metric_result[model_name][j][0],
                    '%.2f'%(metric_result[model_name][j][0]), fontsize=5)
            line = ax.errorbar(nfeats_choice[:i+1],
                [metric_result[model_name][j][0] for j in range(i+1)],
                yerr=[metric_result[model_name][j][1] for j in range(i+1)],
                fmt='-o', label=model_name)
        ax.set_xlim([min(nfeats_choice)-10, max(nfeats_choice)+10])
        ax.set_ylim([minv-(maxv-minv)*0.15,maxv+(maxv-minv)*0.45])
        plt.title(metric_name, fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper left', ncol=2, fancybox=True)
        plt.xlabel('Number of Features', fontsize=13)
        plt.ylabel(en, fontsize=13)
        plt.savefig('plots/'+en.lower()+'_penalty=%.2f'%(penalty)+'.png')