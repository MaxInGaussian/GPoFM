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

BEST_MODEL_PATH = 'mnist.pkl'


############################ Prior Setting ############################
use_models = ['GPoFF', 'GPoAF', 'GPoHF']
reps = 1
penalty = 1.
feats_num = 5
feats_base = 10
nfeats_range = [feats_base, feats_num*feats_base]
nfeats_length = nfeats_range[1]-nfeats_range[0]
nfeats_choice = [nfeats_range[0]+(i*nfeats_length)//(feats_num-1)
    for i in range(feats_num)]
plot_metric = 'mse'
select_params_metric = 'nmse'
select_model_metric = 'score'
fig = plt.figure(facecolor='white', dpi=120)
visualizer = None
visualizer = Visualizer(fig, plot_metric)
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
    'cv_nfolds': 4,
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
def debug(local):
    locals().update(local)
    print('Debug Commands:')
    while True:
        cmd = input('>>> ')
        if(cmd == ''):
            break
        try:
            exec(cmd)
        except Exception as e:
            import traceback
            traceback.print_tb(e.__traceback__)
    
def plot_dist(*args):
    import seaborn as sns
    for x in args:
        plt.figure()
        sns.distplot(x)
    plt.show()

def load_data(prop=.8):
    from sklearn import datasets
    digits = datasets.load_digits()
    X = digits.data
    ty = digits.target[:, None]
    lbls = np.sort(np.unique(ty))
    y = np.zeros((ty.shape[0], lbls.shape[0]))
    y[np.arange(ty.shape[0]), np.where(ty==lbls)[1]] = 1
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
        for round in range(reps):
            model = GPoFM(ModelClass(nfeats=nfeats, penalty=penalty))
            model.optimize(X_train, y_train, funcs, visualizer, **opt_params)
            if(funcs is None):
                funcs = model.get_compiled_funcs()
            if(not os.path.exists(BEST_MODEL_PATH)):
                model.save(BEST_MODEL_PATH)
                visualizer.train_plot()(-1)
            else:
                best_model = GPoFM(Model().load(BEST_MODEL_PATH))
                best_model.fit(X_train, y_train)
                best_model.score(X_test, y_test)
                model.score(X_test, y_test)
                best_model._print_evals_comparison(model.evals)
                if(model.evals[select_model_metric][1][-1] <
                    best_model.evals[select_model_metric][1][-1]):
                    visualizer.train_plot()(-1)
                    model.save(BEST_MODEL_PATH)
                    print("!"*80)
                    print("!"*30, "NEW BEST PREDICTOR", "!"*30)
                    print("!"*80)
            if(round >= reps//2):
                for res in evals.keys():
                    results[res].append(model.evals[res][1][-1])
        for en in evals.keys():
            eval = (np.mean(results[en]), np.std(results[en]))
            evals[en][1][model_name].append(eval)

############################ Plot Performances ############################
    import os
    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig = plt.figure(facecolor='white', dpi=120)
    ax = fig.add_subplot(111)
    for en, (metric_name, metric_result) in evals.items():
        maxv, minv = 0, 1e5
        for model_name in metric_result.keys():
            for j in range(i+1):
                maxv = max(maxv, metric_result[model_name][j][0]+\
                    metric_result[model_name][j][1])
                minv = min(minv, metric_result[model_name][j][0]-\
                    metric_result[model_name][j][1])
            line = ax.errorbar(nfeats_choice[:i+1],
                [metric_result[model_name][j][0] for j in range(i+1)],
                yerr=[metric_result[model_name][j][1] for j in range(i+1)],
                fmt='o', capsize=6, label=model_name, alpha=0.6)
        ax.set_xlim([min(nfeats_choice)-10, max(nfeats_choice)+10])
        ax.set_ylim([minv-(maxv-minv)*0.2,maxv+(maxv-minv)*0.2])
        ax.grid(True)
        plt.title(metric_name, fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', ncol=3, fancybox=True)
        plt.xlabel('Number of Features', fontsize=13)
        plt.ylabel(en, fontsize=13)
        plt.savefig('plots/'+en.lower()+'.png')
        ax.cla()