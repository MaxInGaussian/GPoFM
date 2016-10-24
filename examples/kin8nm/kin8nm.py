################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import matplotlib.pyplot as plt
try:
    from SCFGP import SCFGP
except:
    print("SCFGP is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../../")
    from SCFGP import SCFGP
    print("done.")

def load_kin8nm_data(proportion=3192./8192):
    from sklearn import datasets
    from sklearn import cross_validation
    kin8nm = datasets.fetch_mldata('regression-datasets kin8nm')
    X, y = kin8nm.data[:, :-1], kin8nm.data[:, -1]
    y = y[:, None]
    X = X.astype(np.float64)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=proportion)
    return X_train, y_train, X_test, y_test

trials_per_model = 50
X_train, y_train, X_test, y_test = load_kin8nm_data()
feature_size_choices = [int(X_train.shape[0]**0.5*(i+1)/3.) for i in range(10)]
metrics = {
    "SCORE": ["Model Selection Score", []],
    "COST": ["Hyperparameter Selection Cost", []],
    "MAE": ["Mean Absolute Error", []],
    "MSE": ["Mean Square Error", []],
    "RMSE": ["Root Mean Square Error", []],
    "NMSE": ["Normalized Mean Square Error", []],
    "MNLP": ["Mean Negative Log Probability", []],
    "TIME(s)": ["Training Time", []],
}
try:
    best_model = SCFGP(verbose=False)
    best_model.load("best_model.pkl")
    best_model_score = best_model.SCORE
except (FileNotFoundError, IOError):
    best_model = None
for feature_size in feature_size_choices:
    funcs = None
    results = {en:[] for en in metrics.keys()}
    for round in range(trials_per_model):
        X_train, y_train, X_test, y_test = load_kin8nm_data()
        model = SCFGP(-1, feature_size, verbose=False)
        if(funcs is None):
            model.fit(X_train, y_train, X_test, y_test)
            funcs = (model.train_func, model.pred_func)
        else:
            model.fit(X_train, y_train, X_test, y_test, funcs)
        if(best_model is None):
            model.save("best_model.pkl")
            best_model = model
        else:
            best_model.predict(X_test, y_test)
            if(model.SCORE > best_model.SCORE):
                model.save("best_model.pkl")
                best_model = model
        results["SCORE"].append(model.SCORE)
        results["COST"].append(model.COST)
        results["MAE"].append(model.TsMAE)
        results["MSE"].append(model.TsMSE)
        results["RMSE"].append(model.TsRMSE)
        results["NMSE"].append(model.TsNMSE)
        results["MNLP"].append(model.TsMNLP)
        results["TIME(s)"].append(model.TrTime)
        print("\n>>>", model.NAME, np.mean(results["SCORE"]), np.mean(results["COST"]))
        print("    Model Selection Score\t\t\t= %.4f%s| Best = %.4f"%(
            model.SCORE, "  ", best_model.SCORE))
        print("    Hyperparameter Selection Cost\t= %.4f%s| Best = %.4f"%(
            model.COST, "  ", best_model.COST))
        print("    Mean Absolute Error\t\t\t\t= %.4f%s| Best = %.4f"%(
            model.TsMAE, "  ", best_model.TsMAE))
        print("    Mean Square Error\t\t\t\t= %.4f%s| Best = %.4f"%(
            model.TsMSE, "  ", best_model.TsMSE))
        print("    Root Mean Square Error\t\t\t= %.4f%s| Best = %.4f"%(
            model.TsRMSE, "  ", best_model.TsRMSE))
        print("    Normalized Mean Square Error\t= %.4f%s| Best = %.4f"%(
            model.TsNMSE, "  ", best_model.TsNMSE))
        print("    Mean Negative Log Probability\t= %.4f%s| Best = %.4f"%(
            model.TsMNLP, "  ", best_model.TsMNLP))
        print("    Training Time\t\t\t\t\t= %.4f%s| Best = %.4f"%(
            model.TrTime, "  ", best_model.TrTime))
    for en in metrics.keys():
        metrics[en][1].append((np.mean(results[en]), np.std(results[en])))

import os
if not os.path.exists('plots'):
    os.mkdir('plots')
for en, (metric_name, metric_result) in metrics.items():
    f = plt.figure(figsize=(8, 6), facecolor='white', dpi=120)
    ax = f.add_subplot(111)
    maxv, minv = 0, 1e5
    for i in range(len(feature_size_choices)):
        maxv = max(maxv, metric_result[i][0])
        minv = min(minv, metric_result[i][0])
        ax.text(feature_size_choices[i], metric_result[i][0], '%.2f' % (
            metric_result[i][0]), fontsize=5)
    line = ax.errorbar(feature_size_choices, [metric_result[i][0] for i in
        range(len(feature_size_choices))], fmt='-o')
    ax.set_xlim([min(feature_size_choices)-10, max(feature_size_choices)+10])
    ax.set_ylim([minv-(maxv-minv)*0.15,maxv+(maxv-minv)*0.45])
    plt.title(metric_name, fontsize=18)
    plt.xlabel('Number of Fourier Features', fontsize=13)
    plt.ylabel(en, fontsize=13)
    plt.savefig('plots/'+en.lower()+'.png')