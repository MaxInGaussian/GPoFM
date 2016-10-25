"""
GPoFM: Gaussian Process Training with
       Optimized Feature Maps for Shift-Invariant Kernels
Github: https://github.com/MaxInGaussian/GPoFM
Author: Max W. Y. Lam [maxingaussian@gmail.com]
"""

try:
    install_instr = """
Please make sure you install a recent enough version of Theano. Note that a
simple 'pip install theano' will usually give you a version that is too old
for Lasagne. See the installation docs for more details:
http://lasagne.readthedocs.org/en/latest/user/installation.html#theano"""
    import theano
    theano.config.mode = 'FAST_RUN'
    theano.config.optimizer = 'fast_run'
    theano.config.reoptimize_unpickled_function = False
except ImportError:
    raise ImportError("Could not import Theano." + install_instr)
else:
    try:
        import theano.tensor.signal.pool
    except ImportError:
        raise ImportError("Your Theano version is too old." + install_instr)
    del install_instr
    del theano

from .utils import *
from .models import *

__version__ = "0.0.9"