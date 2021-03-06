"""
GPoFM: Gaussian Process Training with
       Optimized Feature Maps for Shift-Invariant Kernels
Github: https://github.com/MaxInGaussian/GPoFM
Author: Max W. Y. Lam [maxingaussian@gmail.com]
"""

import os, sys
import pandas as pd
import seaborn as sns
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from GPoFM import *

BEST_MODEL_PATH = 'pdt.pkl'

############################ Prior Setting ############################
use_models = ['GPoMax', 'GPoReLU', 'GPoTanh']
reps = 50
penalty = 0.5
dropout = 1.
transform = True
feats_num = 5
feats_base = 10
nfeats_range = [feats_base, feats_num*feats_base]
nfeats_length = nfeats_range[1]-nfeats_range[0]
nfeats_choice = [nfeats_range[0]+(i*nfeats_length)//(feats_num-1)
    for i in range(feats_num)]
plot_metric = 'mae'
select_params_metric = 'obj'
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
    'dropout': dropout,
    'cv_nfolds': 3,
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

def load_data(prop=0.7):
    FEATURE_HEADER_PATH = 'features.csv'
    SUBJECT_LIST_PATH = 'pdt.csv'
    from sklearn import datasets
    from sklearn import cross_validation
    sns.set(font_scale=2)
    subjects_df = pd.read_csv(SUBJECT_LIST_PATH, index_col=0, header=0)
    subjects_df = subjects_df[np.isfinite(subjects_df['MoCA Total'])]
    sample_size = len(subjects_df)
    y = subjects_df['MoCA Total'].as_matrix()[:, None]
    feats_header = pd.read_csv(FEATURE_HEADER_PATH, index_col=0, header=0)
    feats = feats_header.index.tolist()
    X = subjects_df[feats].as_matrix()
    n_cols = 2
    n_rows = 15
    for i in range(n_rows):
        fg, ax = plt.subplots(nrows=1, ncols=n_cols, figsize=(12, 8))
        for j in range(n_cols):
            sns.violinplot(y=feats[i*n_cols+j], data=subjects_df, ax=ax[j])
            ax[j].set(ylabel=feats_header.loc[feats[i*n_cols+j]]['Feature Description'])
        plt.savefig('plots/'+str(i)+'.png')
    ntrain = y.shape[0]
    train_inds = npr.choice(range(ntrain), int(prop*ntrain), replace=False)
    test_inds = np.setdiff1d(range(ntrain), train_inds)
    X_train, y_train = X[train_inds].copy(), y[train_inds].copy()
    X_test, y_test = X[test_inds].copy(), y[test_inds].copy()
    return X_train, y_train, X_test, y_test

############################ Training Phase ############################
for i, nfeats in enumerate(nfeats_choice):
    for model_name in use_models:
        ModelClass = getattr(sys.modules['GPoFM'], model_name)
        funcs = None
        results = {en:[] for en in evals.keys()}
        for round in range(reps):
            X_train, y_train, X_test, y_test = load_data()
            model = GPoFM(ModelClass(nfeats, penalty, transform))
            model.optimize(X_train, y_train, funcs, visualizer, **opt_params)
            if(funcs is None):
                funcs = model.get_compiled_funcs()
            if(not os.path.exists(BEST_MODEL_PATH)):
                model.save(BEST_MODEL_PATH)
            else:
                best_model = GPoFM(Model().load(BEST_MODEL_PATH))
                best_model.fit(X_train, y_train)
                best_model.score(X_test, y_test)
                model.score(X_test, y_test)
                best_model._print_evals_comparison(model.evals)
                if(model.evals[select_model_metric][1][-1] <
                    best_model.evals[select_model_metric][1][-1]):
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
        ax.set_ylim([minv-abs(maxv-minv)*0.2,maxv+abs(maxv-minv)*0.2])
        ax.grid(True)
        plt.title(metric_name, fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', ncol=3, fancybox=True)
        plt.xlabel('Optimized Features', fontsize=13)
        plt.ylabel(en, fontsize=13)
        plt.savefig('plots/'+en.lower()+'.png')
        ax.cla()