import numpy as np
import matplotlib.pyplot as plt
from theano import shared as Ts, function as Tf, tensor as TT
from theano.sandbox import linalg as Tlin

from .. import __init__

__all__ = [
    "Model",
]

class Model(object):
    
    """
    The :class:`Model` class implemented handy functions shared by all machine
    learning models. It is always called as a subclass for any new model.
    
    Parameters
    ----------
    X_scaling : a string
        The pre-scaling method used for inputs of training data
    y_scaling : a string
        The pre-scaling method used for outpus of training data
    verbose : a bool
        An idicator that determines whether printing training message or not
    """

    ID, NAME, verbose = "", "", True
    X_scaler, y_scaler = [None]*2
    M, N, D = -1, -1, -1
    X, y, hyper, Li, alpha, train_func, pred_func = [None]*7
    
    
    def __init__(self, **args):
        X_trans = 'uniform' if 'X_trans' not in args.keys() else args['X_trans']
        y_trans = 'normal' if 'y_trans' not in args.keys() else args['y_trans']
        verbose = False if 'verbose' not in args.keys() else args['verbose']
        self.X_scaler, self.y_scaler = Scaler(X_trans), Scaler(y_trans)
        self.evals = {
            "SCORE": ["Model Selection Score", []],
            "COST": ["Hyperparameter Selection Cost", []],
            "MAE": ["Mean Absolute Error", []],
            "NMAE": ["Normalized Mean Absolute Error", []],
            "MSE": ["Mean Square Error", []],
            "NMSE": ["Normalized Mean Square Error", []],
            "MNLP": ["Mean Negative Log Probability", []],
            "TIME(s)": ["Training Time", []],
        } if 'evals' not in args.keys() else args['evals']
        self.verbose = verbose
        self.generate_ID()
    
    def echo(self, *arg):
        if(self.verbose):
            import sys
            print(" ".join(map(str, arg)))
            sys.stdout.flush()
    
    def generate_ID(self):
        import string
        self.ID = ''.join(chr(npr.choice([ord(c) for c in (
                string.ascii_uppercase+string.digits)])) for _ in range(5))

    def init_params(self):
        raise NotImplementedError
    
    def unpack_params(self, params):
        raise NotImplementedError
    
    def build_theano_models(self, **args):
        raise NotImplementedError
    
    def get_compiled_funcs(self):
        return self.compiled_funcs

    def set_data(self, X, y):
        """
        X: Normally Distributed Inputs
        Y: Normally Distributed Outputs
        """
        self.echo("-"*60, "\nTransforming training data...")
        self.X_scaler.fit(X)
        self.y_scaler.fit(y)
        self.X = self.X_scaler.forward_transform(X)
        self.y = self.y_scaler.forward_transform(y)
        self.echo("done.")
        self.N, self.D = self.X.shape
        if('train_func' not in self.__dict__.keys()):
            self.echo("-"*60, "\nInitializing SCFGP hyperparameters...")
            self.init_params()
            self.echo("done.")
        else:
            cost, self.alpha, self.Li = self.train_func(self.X, self.y)
    
    def minibatches(self, X, y, batchsize, shuffle=True):
        assert len(X) == len(y)
        if(shuffle):
            inds = np.arange(len(X))
            np.random.shuffle(inds)
        for start_ind in range(0, len(X)-batchsize+1, batchsize):
            if shuffle:
                batch = inds[start_ind:start_ind+batchsize]
            else:
                batch = slice(start_ind, start_ind+batchsize)
            yield X[batch], y[batch]

    def optimize(self, Xv=None, yv=None, funcs=None, visualizer=None, **args):
        obj = 'COST' if 'obj' not in args.keys() else args['obj'].upper()
        obj = 'COST' if obj not in self.evals.keys() else obj
        algo = {'algo': None} if 'algo' not in args.keys() else args['algo']
        nbatches = 1 if 'nbatches' not in args.keys() else args['nbatches']
        batchsize = 150 if 'batchsize' not in args.keys() else args['batchsize']
        cvrg_tol = 1e-4 if 'cvrg_tol' not in args.keys() else args['cvrg_tol']
        max_cvrg = 18 if 'max_cvrg' not in args.keys() else args['max_cvrg']
        max_iter = 500 if 'max_iter' not in args.keys() else args['max_iter']
        if(algo['algo'] not in OPT.algos):
            algo = {
                'algo': 'adam',
                'algo_params': {
                    'learning_rate':0.01,
                    'beta1':0.9,
                    'beta2':0.999,
                    'epsilon':1e-8
                }
            }
        for metric in self.evals.keys():
            self.evals[metric][1] = []
        if(funcs is None):
            self.echo("-"*50, "\nCompiling SCFGP theano model...")
            self.build_theano_models(algo['algo'], algo['algo_params'])
            self.echo("done.")
        else:
            self.train_func, self.train_iter_func, self.pred_func = funcs
        if(visualizer is not None):
            visualizer.model = self
            animate = visualizer.train_with_plot()
        if(Xv is None or yv is None):
            obj = 'COST'
            self.evals['MAE'][1].append(0)
            self.evals['NMAE'][1].append(0)
            self.evals['MSE'][1].append(0)
            self.evals['NMSE'][1].append(0)
            self.evals['MNLP'][1].append(0)
            self.evals['SCORE'][1].append(0)
        self.min_obj_ind = 0
        train_start_time = time.time()
        min_obj_val, argmin_params, cvrg_iter = np.Infinity, self.params, 0
        for iter in range(max_iter):
            if(nbatches > 1):
                cost_sum, params_list, batch_count = 0, [], 0
                for X, y in self.minibatches(self.X, self.y, batchsize):
                    params_list.append(self.params.get_value())
                    cost, self.alpha, self.Li = self.train_iter_func(X, y)
                    cost_sum += cost;batch_count += 1
                    if(batch_count == nbatches):
                        break
                self.params = Ts(np.median(np.array(params_list), axis=0))
                self.evals['COST'][1].append(np.double(cost_sum/batch_count))
            else:
                cost, self.alpha, self.Li = self.train_iter_func(self.X, self.y)
                self.evals['COST'][1].append(cost)
            self.evals['TIME(s)'][1].append(time.time()-train_start_time)
            if(Xv is not None and yv is not None):
                self.predict(Xv, yv)
            if(iter%(max_iter//10) == 1):
                self.echo("-"*17, "VALIDATION ITERATION", iter, "-"*17)
                self._print_current_evals()
            if(visualizer is not None):
                animate(iter)
                plt.pause(0.05)
            obj_val = self.evals[obj][1][-1]
            if(obj_val < min_obj_val):
                if(min_obj_val-obj_val < cvrg_tol):
                    cvrg_iter += 1
                else:
                    cvrg_iter = 0
                min_obj_val = obj_val
                self.min_obj_ind = len(self.evals['COST'][1])-1
                argmin_params = self.params.copy()
            else:
                cvrg_iter += 1
            if(iter > 30 and cvrg_iter > max_cvrg):
                break
            elif(cvrg_iter > max_cvrg*0.5):
                randp = np.random.rand()*cvrg_iter/max_cvrg*0.5
                self.params = (1-randp)*self.params+randp*argmin_params
        self.params = argmin_params.copy()
        cost, self.alpha, self.Li = self.train_func(self.X, self.y)
        self.evals['COST'][1].append(np.double(cost))
        self.evals['TIME(s)'][1].append(time.time()-train_start_time)
        if(Xv is not None and yv is not None):
            self.predict(Xv, yv)
        self.min_obj_ind = len(self.evals['COST'][1])-1
        disp = self.verbose
        self.verbose = True
        self.echo("-"*19, "OPTIMIZATION RESULT", "-"*20)
        self._print_current_evals()
        self.echo("-"*60)
        self.verbose = disp

    def predict(self, Xs, ys=None):
        self.Xs = self.X_scaler.forward_transform(Xs)
        mu_f, std_f = self.pred_func(self.Xs, self.alpha, self.Li)
        mu_y = self.y_scaler.backward_transform(mu_f)
        up_bnd_y = self.y_scaler.backward_transform(mu_f+std_f[:, None])
        dn_bnd_y = self.y_scaler.backward_transform(mu_f-std_f[:, None])
        std_y = 0.5*(up_bnd_y-dn_bnd_y)
        if(ys is not None):
            self.evals['MAE'][1].append(np.mean(np.abs(mu_y-ys)))
            self.evals['NMAE'][1].append(self.evals['MAE'][1][-1]/np.std(ys))
            self.evals['MSE'][1].append(np.mean((mu_y-ys)**2.))
            self.evals['NMSE'][1].append(self.evals['MSE'][1][-1]/np.var(ys))
            self.evals['MNLP'][1].append(0.5*np.mean(((
                ys-mu_y)/std_y)**2+np.log(2*np.pi*std_y**2)))
            self.evals['SCORE'][1].append(
                self.evals['NMSE'][1][-1]/(1+np.exp(-self.evals['MNLP'][1][-1])))
        return mu_y, std_y

    def save(self, path):
        import pickle
        save_vars = ['ID', 'M', 'X_scaler', 'y_scaler', 'train_func',
            'pred_func', 'params', 'alpha', 'Li', 'evals']
        save_dict = {varn: self.__dict__[varn] for varn in save_vars}
        with open(path, "wb") as save_f:
            pickle.dump(save_dict, save_f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        import pickle
        with open(path, "rb") as load_f:
            load_dict = pickle.load(load_f)
        for varn, var in load_dict.items():
            self.__dict__[varn] = var
        self.NAME = "SCFGP (Sparsity=%d, Fourier Features=%d)"%(self.S, self.M)

    def _print_current_evals(self):
        for metric in sorted(self.evals.keys()):
            if(len(self.evals[metric][1]) < len(self.evals['COST'][1])):
                continue
            best_perform_eval = self.evals[metric][1][self.min_obj_ind]
            self.echo(self.NAME, "%7s = %.4e"%(metric, best_perform_eval))




