"""
GPoFM: Gaussian Process Training with
       Optimized Feature Maps for Shift-Invariant Kernels
Github: https://github.com/MaxInGaussian/GPoFM
Author: Max W. Y. Lam [maxingaussian@gmail.com]
"""

import sys, os, string, time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from theano import shared as Ts, function as Tf, tensor as TT

from .. import *

__all__ = [
    'GPoFM',
    'Model',
]

class GPoFM(object):
    
    '''
    The :class:`GPoFM` class is a wrapper for all implemented models that are
    based on optimized feature maps for shift-invariant kernels.
    
    Parameters
    ----------
    model : a :class:`Model` instance
        Model to be wrapped
    '''
    
    setting = None
    
    def __init__(self, model):
        self.setting = model.setting
        model_name = self.setting['id'].split('-')[0]
        WrapClass = getattr(sys.modules['GPoFM'], model_name)
        self.__class__ = type('GPoFM', (WrapClass,), dict(self.__dict__)) 
        for varn, var in model.__dict__.items():
            self.__dict__[varn] = var
    
class Model(object):
    
    '''
    The :class:`Model` class implemented handy functions shared by all machine
    learning models. It is always called as a subclass for any new model.
    
    Parameters
    ----------
    X_trans : a string
        Transformation method used for inputs of training data
    y_trans : a string
        Transformation method used for outpus of training data
    verbose : a bool
        Idicator that determines whether printing training message or not
    '''

    setting, verbose, evals_ind, M, N, D = {'id':''}, False, -1, -1, -1, -1
    X, y, X_Trans, y_Trans, params, compiled_funcs, trained_mats = [None]*7
    
    def __init__(self, **args):
        Xt = 'auto-uniform' if 'X_trans' not in args.keys() else args['X_trans']
        yt = 'auto-normal' if 'y_trans' not in args.keys() else args['y_trans']
        verbose = False if 'verbose' not in args.keys() else args['verbose']
        self.trans = {'X': Transformer(Xt), 'y': Transformer(yt)}
        self.verbose = verbose
        self.generate_instance_identifier()
        self.evals = {
            'score': ['Model Selection Score', []],
            'obj': ['Params Optimization Objective', []],
            'mae': ['Mean Absolute Error', []],
            'nmae': ['Normalized Mean Absolute Error', []],
            'mse': ['Mean Square Error', []],
            'nmse': ['Normalized Mean Square Error', []],
            'mnlp': ['Mean Negative Log Probability', []],
            'time': ['Training Time(s)', []],
        }
    
    def __str__(self):
        raise NotImplementedError

    def init_params(self):
        raise NotImplementedError
    
    def unpack_params(self, params):
        raise NotImplementedError
    
    def unpack_trained_mats(self, trained_mats):
        raise NotImplementedError
    
    def unpack_predicted_mats(self, predicted_mats):
        raise NotImplementedError
    
    def pack_train_func_inputs(self, X, y):
        raise NotImplementedError
    
    def pack_pred_func_inputs(self, Xs):
        raise NotImplementedError

    def pack_save_vars(self):
        raise NotImplementedError
    
    def compile_theano_funcs(self, opt_algo, opt_params):
        raise NotImplementedError
    
    def echo(self, *arg):
        if(self.verbose):
            print(' '.join(map(str, arg)))
            sys.stdout.flush()
    
    def generate_instance_identifier(self):
        self.setting = {'id': self.__class__.__name__+'-'+''.join(
            chr(npr.choice([ord(c) for c in (
                string.ascii_uppercase+string.digits)])) for _ in range(5))}
    
    def get_compiled_funcs(self):
        return self.compiled_funcs
    
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

    def set_data(self, X, y):
        '''
        X: Normally Distributed Inputs
        Y: Normally Distributed Outputs
        '''
        self.echo('-'*80, '\nTransforming training data...')
        self.X = self.trans['X'].fit_transform(X)
        self.y = self.trans['y'].fit_transform(y)
        self.echo('done.')
        self.N, self.D = self.X.shape
        if(self.trained_mats is None):
            self.echo('-'*80, '\nInitializing hyperparameters...')
            self.init_params()
            self.echo('done.')
        else:
            trained_mats = self.compiled_funcs['train'](self.X, self.y)
            self.trained_mats = self.unpack_trained_mats(trained_mats)

    def optimize(self, Xv=None, yv=None, funcs=None, visualizer=None, **args):
        obj_type = 'obj' if 'obj' not in args.keys() else args['obj'].lower()
        obj_type = 'obj' if obj_type not in self.evals.keys() else obj_type
        opt_algo = {'algo': None} if 'algo' not in args.keys() else args['algo']
        nbatches = 1 if 'nbatches' not in args.keys() else args['nbatches']
        batchsize = 150 if 'batchsize' not in args.keys() else args['batchsize']
        cvrg_tol = 1e-4 if 'cvrg_tol' not in args.keys() else args['cvrg_tol']
        max_cvrg = 18 if 'max_cvrg' not in args.keys() else args['max_cvrg']
        max_iter = 500 if 'max_iter' not in args.keys() else args['max_iter']
        if(opt_algo['algo'] not in Optimizer.algos):
            opt_algo = {
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
            self.echo('-'*80, '\nCompiling theano functions...')
            self.compile_theano_funcs(opt_algo['algo'], opt_algo['algo_params'])
            self.echo('done.')
        else:
            self.compiled_funcs = funcs
        if(visualizer is not None):
            visualizer.model = self
            animate = visualizer.train_with_plot()
        if(Xv is None or yv is None):
            obj_type = 'obj'
            self.evals['mae'][1].append(np.Infinity)
            self.evals['nmae'][1].append(np.Infinity)
            self.evals['mse'][1].append(np.Infinity)
            self.evals['nmse'][1].append(np.Infinity)
            self.evals['mnlp'][1].append(np.Infinity)
            self.evals['score'][1].append(np.Infinity)
        self.evals_ind = 0
        train_start_time = time.time()
        min_obj_val, argmin_params, cvrg_iter = np.Infinity, self.params, 0
        for iter in range(max_iter):
            if(nbatches > 1):
                obj_sum, params_list, batch_count = 0, [], 0
                for X, y in self.minibatches(self.X, self.y, batchsize):
                    params_list.append(self.params.get_value())
                    train_inputs = self.pack_train_func_inputs(self.X, self.y)
                    trained_mats = self.compiled_funcs['opt'](*train_inputs)
                    self.trained_mats = self.unpack_trained_mats(trained_mats)
                    obj_sum += self.trained_mats['obj'];batch_count += 1
                    if(batch_count == nbatches):
                        break
                self.params = Ts(np.median(np.array(params_list), axis=0))
                self.evals['obj'][1].append(np.double(obj_sum/batch_count))
            else:
                train_inputs = self.pack_train_func_inputs(self.X, self.y)
                trained_mats = self.compiled_funcs['opt'](*train_inputs)
                self.trained_mats = self.unpack_trained_mats(trained_mats)
                self.evals['obj'][1].append(self.trained_mats['obj'])
            self.evals['time'][1].append(time.time()-train_start_time)
            if(Xv is not None and yv is not None):
                self.predict(Xv, yv)
            if(iter%(max_iter//10) == 1):
                self.echo('-'*26, 'VALIDATION ITERATION', iter, '-'*27)
                self._print_current_evals()
            if(visualizer is not None):
                animate(iter)
                plt.pause(0.05)
            obj_val = self.evals[obj_type][1][-1]
            if(obj_val < min_obj_val):
                if(min_obj_val-obj_val < cvrg_tol):
                    cvrg_iter += 1
                else:
                    cvrg_iter = 0
                min_obj_val = obj_val
                self.evals_ind = len(self.evals['obj'][1])-1
                argmin_params = self.params.copy()
            else:
                cvrg_iter += 1
            if(iter > 30 and cvrg_iter > max_cvrg):
                break
            elif(cvrg_iter > max_cvrg*0.5):
                randp = np.random.rand()*cvrg_iter/max_cvrg*0.5
                self.params = (1-randp)*self.params+randp*argmin_params
        self.params = argmin_params.copy()
        train_inputs = self.pack_train_func_inputs(self.X, self.y)
        trained_mats = self.compiled_funcs['train'](*train_inputs)
        self.trained_mats = self.unpack_trained_mats(trained_mats)
        self.evals['obj'][1].append(self.trained_mats['obj'])
        self.evals['time'][1].append(time.time()-train_start_time)
        if(Xv is not None and yv is not None):
            self.predict(Xv, yv)
        for metric in self.evals.keys():
            self.evals[metric][1] = [self.evals[metric][1][-1]]
        self.evals_ind = -1
        verbose = self.verbose
        self.verbose = True
        self.echo('-'*29, 'OPTIMIZATION RESULT', '-'*30)
        self._print_current_evals()
        self.verbose = verbose

    def predict(self, Xs, ys=None):
        self.Xs = self.trans['X'].transform(Xs)
        pred_inputs = self.pack_pred_func_inputs(self.Xs)
        predicted_mats = self.compiled_funcs['pred'](*pred_inputs)
        self.predicted_mats = self.unpack_predicted_mats(predicted_mats)
        mu_fs = self.predicted_mats['mu_fs']
        std_fs = self.predicted_mats['std_fs']
        mu_ys = self.trans['y'].recover(mu_fs)
        up_bnd_ys = self.trans['y'].recover(mu_fs+std_fs[:, None])
        dn_bnd_ys = self.trans['y'].recover(mu_fs-std_fs[:, None])
        std_ys = 0.5*(up_bnd_ys-dn_bnd_ys)
        if(ys is not None):
            y = self.trans['y'].recover(self.y)
            mu_y = self.trans['y'].recover(self.trained_mats['mu_f'])
            diff_y = np.concatenate([mu_ys-ys, mu_y-y])
            mae = np.mean(np.abs(diff_y))
            self.evals['mae'][1].append(mae)
            nmae = mae/np.std(ys)
            self.evals['nmae'][1].append(nmae)
            mse = np.mean(diff_y**2.)
            self.evals['mse'][1].append(mse)
            nmse = mse/np.var(ys)
            self.evals['nmse'][1].append(nmse)
            mnlp = 0.5*np.mean(((ys-mu_ys)/std_ys)**2+np.log(2*np.pi*std_ys**2))
            self.evals['mnlp'][1].append(mnlp)
            score = nmse/(1+np.exp(-mnlp))
            self.evals['score'][1].append(score)
        return mu_ys, std_ys

    def get_vars_for_prediction(self):
        return ['setting',
                'trans',
                'params',
                'trained_mats',
                'compiled_funcs',
                'evals',
                'y']

    def save(self, path):
        import pickle
        save_vars = self.get_vars_for_prediction()
        save_dict = {varn: self.__dict__[varn] for varn in save_vars}
        sys.setrecursionlimit(10000)
        with open(path, 'wb') as save_f:
            pickle.dump(save_dict, save_f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        import pickle
        with open(path, 'rb') as load_f:
            load_dict = pickle.load(load_f)
        for varn, var in load_dict.items():
            self.__dict__[varn] = var
        return self

    def _print_current_evals(self):
        for metric in sorted(self.evals.keys()):
            eval = self.evals[metric][1][self.evals_ind]
            model_name = self.__str__()
            float_len = 64-len(model_name) if eval > 0 else 63-len(model_name)
            aligned = ('%6s = %.'+str(float_len)+'e')%(metric, eval)
            self.echo(model_name, aligned)

    def _print_evals_comparison(self, evals):
        verbose = self.verbose
        self.verbose = True
        self.echo('-'*30, 'COMPARISON RESULT', '-'*31)
        for metric in sorted(self.evals.keys()):
            eval1 = self.evals[metric][1][self.evals_ind]
            eval2 = evals[metric][1][-1]
            model_name = self.__str__()
            float_len = 27-len(model_name)//2
            float_len1 = float_len-1 if eval1 < 0 else float_len
            float_len2 = float_len-1 if eval2 < 0 else float_len
            aligned = ('%6s = %.'+str(float_len1)+'e <> '+'%.'+str(
                float_len2-len(model_name)%2)+'e')%(metric, eval1, eval2)
            self.echo(model_name, aligned)
        self.verbose = verbose




