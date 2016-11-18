"""
GPoFM: Gaussian Process Training with
       Optimized Feature Maps for Shift-Invariant Kernels
Github: https://github.com/MaxInGaussian/GPoFM
Author: Max W. Y. Lam [maxingaussian@gmail.com]
"""

import sys, os, string, time
import numpy as np
import numpy.random as npr

from theano import shared as Ts, function as Tf, tensor as TT
from theano.sandbox import linalg as Tlin

from .. import Optimizer, Transformer, Visualizer

__all__ = [
    'GPoFM',
    'Model',
]

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

    setting, unfitted, verbose = {'id':''}, True, True
    evals_ind, M, N, D = -1, -1, -1, -1
    X, y, X_Trans, y_Trans, params, compiled_funcs, trained_mats = [None]*7
    
    def __init__(self, nfeats=50, penalty=1., transform=True, **args):
        Xt = 'normal' if 'X_trans' not in args.keys() else args['X_trans']
        yt = 'normal' if 'y_trans' not in args.keys() else args['y_trans']
        verbose = False if 'verbose' not in args.keys() else args['verbose']
        self.trans = {'X': Transformer(Xt), 'y': Transformer(yt)}
        self.verbose = verbose
        rand_str = ''.join(chr(npr.choice([ord(c) for c in (
            string.ascii_uppercase+string.digits)])) for _ in range(5))
        self.setting = {}
        self.setting['id'] = self.__class__.__name__+'-'+rand_str
        self.setting['nfeats'] = nfeats
        self.setting['penalty'] = penalty
        self.setting['transform'] = transform
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

    def randomized_params(self):
        raise NotImplementedError
    
    def feature_maps(self, X, params):
        raise NotImplementedError
    
    def inv_feature_maps(self, Phi, params):
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
    
    def get_params(self, **arg):
        if(self.params is None):
            self.echo('-'*80, '\nInitializing hyperparameters...')
            self.params = Ts(np.concatenate(self.randomized_params()))
            self.echo('done.')
        return self.params.eval()
    
    def get_compiled_funcs(self):
        return self.compiled_funcs

    def fit(self, X, y, update_params=False):
        self.unfitted = False
        self.Xt = self.trans['X'].transform(X)
        self.yt = self.trans['y'].transform(y)
        self.N, self.D = self.Xt.shape
        if(self.params is None):
            self.echo('-'*80, '\nInitializing hyperparameters...')
            self.params = Ts(np.concatenate(self.randomized_params()))
            self.echo('done.')
        else:
            trained_mats = self.compiled_funcs['opt' if update_params else
                'train'](self.Xt, self.yt)
            self.trained_mats = self.unpack_trained_mats(trained_mats)
    
    def unpack_trained_mats(self, trained_mats):
        return {'obj': np.double(trained_mats[0]),
                'alpha': trained_mats[1],
                'Li': trained_mats[2],
                'mu_f': trained_mats[3],}
    
    def unpack_predicted_mats(self, predicted_mats):
        return {'mu_fs': predicted_mats[0],
                'std_fs': predicted_mats[1],}

    def theano_input_data(self, params):
        return TT.dmatrices('X')

    def theano_output_data(self, params, inverse=None):
        if(inverse is not None):
            return inverse
        return TT.dmatrices('y')
    
    def pack_train_func_inputs(self, X, y):
        return [X, y]
    
    def pack_pred_func_inputs(self, Xs):
        return [Xs, self.trained_mats['alpha'], self.trained_mats['Li']]

    def compile_theano_funcs(self, opt_algo, opt_params, dropout):
        self.compiled_funcs = {}
        # Compile Train & Optimization Function
        eps, S = 1e-6, self.setting['nfeats']
        kl = lambda mu, std: TT.mean(std+mu**2-TT.log(std))
        params = TT.dvector('params')
        X = self.theano_input_data(params)
        y = self.theano_output_data(params)
        sig2_n, sig2_f, FF, Phi = self.feature_maps(X, params)
        srng = TT.shared_randomstreams.RandomStreams(npr.randint(888))
        mask = srng.binomial(n=1, p=dropout, size=(1, Phi.shape[1]))
        Phi = Phi*mask
        PhiTPhi = TT.dot(Phi.T, Phi)
        W = (sig2_n+eps)*TT.identity_like(PhiTPhi)
        A = PhiTPhi+W
        L = Tlin.cholesky(A)
        Li = Tlin.matrix_inverse(L)
        PhiTy = Phi.T.dot(y)
        beta = TT.dot(Li, PhiTy)
        alpha = TT.dot(Li.T, beta)*mask.T
        mu_f = TT.dot(Phi, alpha)
        mu_w = TT.mean(FF, axis=1)
        sig_w = TT.std(FF, axis=1)
        gof = 1./sig2_n*((y**2).sum()-(beta**2).sum())
        pnt = TT.sum(2*TT.log(TT.diagonal(L))+TT.log(sig2_f))+kl(mu_w, sig_w)
        obj = (gof+pnt*self.setting['penalty'])/X.shape[0]+TT.log(sig2_n)
        grads = TT.grad(obj, params)
        updates = {self.params: grads}
        updates = getattr(Optimizer, opt_algo)(updates, **opt_params)
        updates = getattr(Optimizer, 'nesterov')(updates, momentum=0.9)
        train_inputs = [X, y]
        train_outputs = [obj, alpha, Li, mu_f]
        self.compiled_funcs['opt'] = Tf(train_inputs, train_outputs,
            givens=[(params, self.params)], updates=updates)
        self.compiled_funcs['train'] = Tf(train_inputs, train_outputs,
            givens=[(params, self.params)])
        # Compile Predict Function
        Li, alpha = TT.dmatrices('Li', 'alpha')
        Xs = self.theano_input_data(params)
        sig2_n, _, _, Phis = self.feature_maps(Xs, params)
        mu_pred = TT.dot(Phis, alpha*dropout)
        std_pred = ((sig2_n*(1+(TT.dot(Phis, Li.T)**2).sum(1)))**0.5)[:, None]
        up_bnd = self.theano_output_data(params, mu_pred+std_pred)
        lw_bnd = self.theano_output_data(params, mu_pred-std_pred)
        mu_pred = self.theano_output_data(params, mu_pred)
        std_pred = 0.5*(up_bnd-lw_bnd)
        pred_inputs = [Xs, alpha, Li]
        pred_outputs = [mu_pred, std_pred]
        self.compiled_funcs['pred'] = Tf(pred_inputs, pred_outputs,
            givens=[(params, self.params)])

    def score(self, X, y):
        self.Xs, self.ys = X.copy(), y.copy()
        mu, std = self.predict(X)
        mae = np.mean(np.abs(mu-y))
        self.evals['mae'][1].append(mae)
        nmae = mae/np.std(y)
        self.evals['nmae'][1].append(nmae)
        mse = np.mean((mu-y)**2.)
        self.evals['mse'][1].append(mse)
        nmse = mse/np.var(y)
        self.evals['nmse'][1].append(nmse)
        mnlp = 0.5*np.mean(((y-mu)/std)**2+np.log(2*np.pi*std**2))
        self.evals['mnlp'][1].append(mnlp)
        score = nmse/(1+np.exp(-mnlp))
        self.evals['score'][1].append(score)
        return score

    def cross_validate(self, X, y, nfolds):
        from sklearn.model_selection import ShuffleSplit
        ss = ShuffleSplit(n_splits=nfolds, random_state=npr.randint(888))
        cv_evals_sum = {metric: [] for metric in self.evals.keys()}
        for train, valid in ss.split(X):
            Xt, yt = X[train], y[train]
            Xv, yv = X[valid], y[valid]
            self.fit(Xt, yt, self.unfitted)
            cv_evals_sum['obj'].append(self.trained_mats['obj'])
            self.score(Xv, yv)
            for metric in self.evals.keys():
                if(metric == 'obj' or metric == 'time'):
                    continue
                cv_evals_sum[metric].append(self.evals[metric][1].pop())
        self.fit(X, y, True)
        cv_evals_sum['time'].append(time.time()-self.train_start_time)
        cv_evals_sum['obj'].append(self.trained_mats['obj'])
        for metric in self.evals.keys():
            self.evals[metric][1].append(np.mean(cv_evals_sum[metric]))

    def optimize(self, X, y, funcs=None, visualizer=None, **args):
        self.trans['X'].fit(X); self.trans['y'].fit(y)
        self.fit(X, y)
        obj_type = 'obj' if 'obj' not in args.keys() else args['obj'].lower()
        obj_type = 'obj' if obj_type not in self.evals.keys() else obj_type
        opt_algo = {'algo': None} if 'algo' not in args.keys() else args['algo']
        cv_nfolds = 5 if 'cv_nfolds' not in args.keys() else args['cv_nfolds']
        cvrg_tol = 1e-4 if 'cvrg_tol' not in args.keys() else args['cvrg_tol']
        max_cvrg = 18 if 'max_cvrg' not in args.keys() else args['max_cvrg']
        max_iter = 500 if 'max_iter' not in args.keys() else args['max_iter']
        dropout = 1. if 'dropout' not in args.keys() else args['dropout']
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
            algo, algo_params = opt_algo['algo'], opt_algo['algo_params']
            self.compile_theano_funcs(algo, algo_params, dropout)
            self.echo('done.')
        else:
            self.compiled_funcs = funcs
        if(visualizer is not None):
            visualizer.model = self
            animate = visualizer.train_plot()
        self.evals_ind = 0
        self.train_start_time = time.time()
        min_obj, min_obj_val = np.Infinity, np.Infinity
        argmin_params, cvrg_iter = self.params, 0
        for iter in range(max_iter):
            self.cross_validate(X, y, cv_nfolds)
            if(iter%(max_iter//10) == 1):
                self.echo('-'*26, 'VALIDATION ITERATION', iter, '-'*27)
                self._print_current_evals()
            if(visualizer is not None):
                animate(iter)
            obj_val = self.evals[obj_type][1][-1]
            if(obj_val < min_obj_val):
                if(min_obj_val-obj_val < cvrg_tol):
                    cvrg_iter += 1
                else:
                    cvrg_iter = 0
                min_obj = self.evals['obj'][1][-1]
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
        self.cross_validate(X, y, cv_nfolds)
        self.evals_ind = -1
        verbose = self.verbose
        self.verbose = True
        self.echo('-'*29, 'OPTIMIZATION RESULT', '-'*30)
        self._print_current_evals()
        self.verbose = verbose

    def predict(self, Xs):
        self.Xs = self.trans['X'].transform(Xs)
        pred_inputs = self.pack_pred_func_inputs(self.Xs)
        predicted_mats = self.compiled_funcs['pred'](*pred_inputs)
        self.predicted_mats = self.unpack_predicted_mats(predicted_mats)
        mu_fs = self.predicted_mats['mu_fs']
        std_fs = self.predicted_mats['std_fs']
        mu_ys = self.trans['y'].recover(mu_fs)
        up_bnd_ys = self.trans['y'].recover(mu_fs+std_fs)
        dn_bnd_ys = self.trans['y'].recover(mu_fs-std_fs)
        std_ys = 0.5*(up_bnd_ys-dn_bnd_ys)
        return mu_ys, std_ys

    def get_vars_for_prediction(self):
        return ['setting',
                'trans',
                'params',
                'trained_mats',
                'compiled_funcs',
                'evals']

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




