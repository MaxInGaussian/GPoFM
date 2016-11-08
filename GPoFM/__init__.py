"""
GPoFM: Gaussian Process Training with
       Optimized Feature Maps for Shift-Invariant Kernels
Github: https://github.com/MaxInGaussian/GPoFM
Author: Max W. Y. Lam [maxingaussian@gmail.com]
"""

__version__ = "0.0.9"

try:
    instr = "Please make sure you install the latest version of Theano."
    import theano
    theano.config.mode = 'FAST_RUN'
    theano.config.optimizer = 'fast_run'
    theano.config.reoptimize_unpickled_function = False
except ImportError:
    raise ImportError("Could not import Theano." + instr)
else:
    try:
        import theano.tensor.signal.pool
    except ImportError:
        raise ImportError("Your Theano version is too old. "+instr)
    del instr
    del theano

from .utils import *
from .models import *