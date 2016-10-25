"""
GPoFM: Gaussian Process Training with
       Optimized Feature Maps for Shift-Invariant Kernels
Github: https://github.com/MaxInGaussian/GPoFM
Author: Max W. Y. Lam [maxingaussian@gmail.com]
"""

import numpy as np
import theano
import theano.tensor as TT
from collections import OrderedDict

__all__ = [
    "Optimizer",
]

class Optimizer(object):

    algos = [
        "apply_momentum",
        "apply_nesterov_momentum",
        "sgd",
        "adagrad",
        "rmsprop",
        "adadelta",
        "adam",
        "adamax",
        "norm_constraint",
        "total_norm_constraint"
    ]
    
    @staticmethod
    def apply_momentum(updates, momentum=0.9):
        """Returns a modified update dictionary including momentum
        Generates update expressions of the form:
    *``velocity := momentum*velocity+updates[param]-param``
    *``param := param+velocity``
        Parameters
        ----------
        updates : OrderedDict
            A dictionary mapping parameters to update expressions
        params : iterable of shared variables, optional
            The variables to apply momentum to. If omitted, will apply
            momentum to all `updates.keys()`.
        momentum : float or symbolic scalar, optional
            The amount of momentum to apply. Higher momentum results in
            smoothing over more update steps. Defaults to 0.9.
        Returns
        -------
        OrderedDict
            A copy of `updates` with momentum updates for all `params`.
        Notes
        -----
        Higher momentum also results in larger update steps. To counter that,
        you can optionally scale your learning rate by `1-momentum`.
        """
        params = list(updates.keys())[0]
        updates = OrderedDict(updates)
        value = params.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                broadcastable=params.broadcastable)
        x = momentum*velocity+updates[params]
        updates[velocity] = x-params
        updates[params] = x
        return updates
    
    @staticmethod
    def apply_nesterov_momentum(updates, momentum=0.9):
        """Returns a modified update dictionary including Nesterov momentum
        Generates update expressions of the form:
        *``velocity := momentum*velocity+updates[params]-params``
        *``params := params+momentum*velocity+updates[params]-params``
        Parameters
        ----------
        updates : OrderedDict
            A dictionary mapping parameters to update expressions
        momentum : float or symbolic scalar, optional
            The amount of momentum to apply. Higher momentum results in
            smoothing over more update steps. Defaults to 0.9.
        Returns
        -------
        OrderedDict
            A copy of `updates` with momentum updates for all `params`.
        Notes
        -----
        Higher momentum also results in larger update steps. To counter that,
        you can optionally scale your learning rate by `1-momentum`.
        The classic formulation of Nesterov momentum (or Nesterov accelerated
        gradient) requires the gradient to be evaluated at the predicted next
        position in parameter space. Here, we use the formulation described at
        https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
        which allows the gradient to be evaluated at the current parameters.
        """
        params = list(updates.keys())[0]
        updates = OrderedDict(updates)
        value = params.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                broadcastable=params.broadcastable)
        x = momentum*velocity+updates[params]-params
        updates[velocity] = x
        updates[params] = momentum*x+updates[params]
        return updates
    
    @staticmethod
    def sgd(params, grads,
        learning_rate=0.01,
        **args):
        """Stochastic Gradient Descent (SGD) updates
        Generates update expressions of the form:
        *``params := params-learning_rate*gradient``
        Parameters
        ----------
        params : theano shared variable
        grads : theano symbolic variable
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        Returns
        -------
        OrderedDict
            A dictionary mapping each parameter to its update expression
        """
        updates = OrderedDict()
        updates[params] = params-learning_rate*grads
        return updates
    
    @staticmethod
    def adagrad(params, grads,
        learning_rate=0.01,
        epsilon=1e-6,
        **args):
        """Adagrad updates
        Scale learning rates by dividing with the square root of accumulated
        squared gradients. See [1]_ for further description.
        Parameters
        ----------
        params : theano shared variable
        grads : theano symbolic variable
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        epsilon : float or symbolic scalar
            Small value added for numerical stability
        Returns
        -------
        OrderedDict
            A dictionary mapping each parameter to its update expression
        Notes
        -----
        Using step size eta Adagrad calculates the learning rate for feature i at
        time step t as:
        .. math:: \\eta_{t,i} = \\frac{\\eta}
        {\\sqrt{\\sum^t_{t^\\prime} g^2_{t^\\prime,i}+\\epsilon}} g_{t,i}
        as such the learning rate is monotonically decreasing.
        Epsilon is not included in the typical formula, see [2]_.
        References
        ----------
        .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
            Adaptive subgradient methods for online learning and stochastic
            optimization. JMLR, 12:2121-2159.
        .. [2] Chris Dyer:
            Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
        """
        updates = OrderedDict()
        value = params.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                            broadcastable=params.broadcastable)
        accu_new = accu+grads**2
        updates[accu] = accu_new
        updates[params] = params-(learning_rate*grads/TT.sqrt(accu_new+epsilon))
        return updates
    
    @staticmethod
    def rmsprop(params, grads,
        learning_rate=0.01,
        rho=0.9,
        epsilon=1e-6,
        **args):
        """RMSProp updates
        Scale learning rates by dividing with the moving average of the root mean
        squared (RMS) gradients. See [1]_ for further description.
        Parameters
        ----------
        params : theano shared variable
        grads : theano symbolic variable
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        rho : float or symbolic scalar
            Gradient moving average decay factor
        epsilon : float or symbolic scalar
            Small value added for numerical stability
        Returns
        -------
        OrderedDict
            A dictionary mapping each parameter to its update expression
        Notes
        -----
        `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
        moving average slowly and a value close to 0 will decay the moving average
        fast.
        Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
        learning rate :math:`\\eta_t` is calculated as:
        .. math::
        r_t &= \\rho r_{t-1}+(1-\\rho)*g^2\\\\
        \\eta_t &= \\frac{\\eta}{\\sqrt{r_t+\\epsilon}}
        References
        ----------
        .. [1] Tieleman, TT. and Hinton, G. (2012):
            Neural Networks for Machine Learning, Lecture 6.5-rmsprop.
            Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
        """
        updates = OrderedDict()
        one = TT.constant(1)
        value = params.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                            broadcastable=params.broadcastable)
        accu_new = rho*accu+(one-rho)*grad**2
        updates[accu] = accu_new
        updates[params] = params-(learning_rate*grads/TT.sqrt(accu_new+epsilon))
        return updates
    
    @staticmethod
    def adadelta(params, grads,
        learning_rate=0.01,
        rho=0.95,
        epsilon=1e-6,
        **args):
        """ Adadelta updates
        Scale learning rates by the ratio of accumulated gradients to accumulated
        updates, see [1]_ and notes for further description.
        Parameters
        ----------
        params : theano shared variable
        grads : theano symbolic variable
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        rho : float or symbolic scalar
            Squared gradient moving average decay factor
        epsilon : float or symbolic scalar
            Small value added for numerical stability
        Returns
        -------
        OrderedDict
            A dictionary mapping each parameter to its update expression
        Notes
        -----
        rho should be between 0 and 1. A value of rho close to 1 will decay the
        moving average slowly and a value close to 0 will decay the moving average
        fast.
        rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to
        work for multiple datasets (MNIST, speech).
        In the paper, no learning rate is considered (so learning_rate=1.0).
        Probably best to keep it at this value.
        epsilon is important for the very first update (so the numerator does
        not become 0).
        Using the step size eta and a decay factor rho the learning rate is
        calculated as:
        .. math::
        r_t &= \\rho r_{t-1}+(1-\\rho)*g^2\\\\
        \\eta_t &= \\eta \\frac{\\sqrt{s_{t-1}+\\epsilon}}
                                {\sqrt{r_t+\epsilon}}\\\\
        s_t &= \\rho s_{t-1}+(1-\\rho)*(\\eta_t*g)^2
        References
        ----------
        .. [1] Zeiler, M. D. (2012):
            ADADELTA: An Adaptive Learning Rate Method.
            arXiv Preprint arXiv:1212.5701.
        """
        updates = OrderedDict()
        one = TT.constant(1)
        value = params.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                            broadcastable=params.broadcastable)
        delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                broadcastable=params.broadcastable)
        accu_new = rho*accu+(one-rho)*grads**2
        updates[accu] = accu_new
        update = (grads*TT.sqrt(delta_accu+epsilon)/
                TT.sqrt(accu_new+epsilon))
        updates[params] = params-learning_rate*update
        delta_accu_new = rho*delta_accu+(one-rho)*update**2
        updates[delta_accu] = delta_accu_new
        return updates
    
    @staticmethod
    def adam(params, grads,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.99,
        epsilon=1e-8,
        **args):
        """Adam updates
        Adam updates implemented as in [1]_.
        Parameters
        ----------
        params : theano shared variable
        grads : theano symbolic variable
        learning_rate : float
            Learning rate
        beta1 : float
            Exponential decay rate for the first moment estimates.
        beta2 : float
            Exponential decay rate for the second moment estimates.
        epsilon : float
            Constant for numerical stability.
        Returns
        -------
        OrderedDict
            A dictionary mapping each parameter to its update expression
        Notes
        -----
        The paper [1]_ includes an additional hyperparameter lambda. This is only
        needed to prove convergence of the algorithm and has no practical use
        (personal communication with the authors), it is therefore omitted here.
        References
        ----------
        .. [1] Kingma, Diederik, and Jimmy Ba (2014):
            Adam: A Method for Stochastic Optimization.
            arXiv preprint arXiv:1412.6980.
        """
        t_prev = theano.shared(np.asarray(0., dtype=theano.config.floatX))
        updates = OrderedDict()
        one = TT.constant(1)
        t = t_prev+1
        a_t = learning_rate*TT.sqrt(one-beta2**t)/(one-beta1**t)
        value = params.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                            broadcastable=params.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                            broadcastable=params.broadcastable)
        m_t = beta1*m_prev+(one-beta1)*grads
        v_t = beta2*v_prev+(one-beta2)*grads**2
        step = a_t*m_t/(TT.sqrt(v_t)+epsilon)
        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[params] = params-step
        updates[t_prev] = t
        return updates
    
    @staticmethod
    def adamax(params, grads,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        **args):
        """Adamax updates
        Adamax updates implemented as in [1]_. This is a variant of of the Adam
        algorithm based on the infinity norm.
        Parameters
        ----------
        params : theano shared variable
        grads : theano symbolic variable
        learning_rate : float
            Learning rate
        beta1 : float
            Exponential decay rate for the first moment estimates.
        beta2 : float
            Exponential decay rate for the weighted infinity norm estimates.
        epsilon : float
            Constant for numerical stability.
        Returns
        -------
        OrderedDict
            A dictionary mapping each parameter to its update expression
        References
        ----------
        .. [1] Kingma, Diederik, and Jimmy Ba (2014):
            Adam: A Method for Stochastic Optimization.
            arXiv preprint arXiv:1412.6980.
        """
        t_prev = theano.shared(np.asarray(0., dtype=theano.config.floatX))
        updates = OrderedDict()
        one = TT.constant(1)
        t = t_prev+1
        a_t = learning_rate/(one-beta1**t)
        value = params.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                            broadcastable=params.broadcastable)
        u_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                            broadcastable=params.broadcastable)
        m_t = beta1*m_prev+(one-beta1)*grads
        u_t = TT.maximum(beta2*u_prev, abs(grads))
        step = a_t*m_t/(u_t+epsilon)
        updates[m_prev] = m_t
        updates[u_prev] = u_t
        updates[params] = params-step
        updates[t_prev] = t
        return updates
