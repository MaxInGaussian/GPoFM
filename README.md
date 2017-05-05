# GPoFM

Gaussian Process Training with Optimized Feature Maps for Shift-Invariant Kernel

## Installation

To install GPoFM, clone this repo:

    $ git clone https://github.com/MaxInGaussian/GPoFM.git
    $ python setup.py install

## Train GPoFM with Different Feature Mapping
```python
from GPoFM import *
use_models = ['GPoMax', 'GPoReLU', 'GPoTanh']
for model_name in use_models:
    ModelClass = getattr(sys.modules['GPoFM'], model_name)
    model = GPoFM(ModelClass(nfeats, resolution, penalty, transform))
    model.optimize(X_train, y_train, funcs, visualizer, **opt_params)
```

## Find the Tailored Trained GPoFM Model
```python
# fit current dataset to original best trained model
best_model = GPoFM(Model().load(BEST_MODEL_PATH))
best_model.fit(X_train, y_train)
best_model.score(X_test, y_test)
# compare a new trained to original best trained model
model.score(X_test, y_test)
if(model.evals['score'][1][-1] < best_model.evals['score'][1][-1]):
    # save if the new model gives better score
    model.save('best_model.pkl')
```

# Performances on Benchmark Regression Datasets
| Benchmark Dataset | Number of Attributes | Size of Training Data | Size of Testing Data |
| :---: | :---: | :---: | :---: |
| Bostion Housing | 13 | 400 | 106 |
| Abalone | 10 | 3133 | 1044 |
| Kin8nm | 10 | 5000 | 3192 |

<h2 align="left">
Bostion Housing
</h2>
<img src ="examples/boston_housing/plots/mae.png" />
<img src ="examples/boston_housing/plots/mse.png" />
<img src ="examples/boston_housing/plots/nmse.png" />
<img src ="examples/boston_housing/plots/mnlp.png" />
<img src ="examples/boston_housing/plots/time.png" />
<h2 align="left">
Abalone
</h2>
<img src ="examples/abalone/plots/mae.png" />
<img src ="examples/abalone/plots/mse.png" />
<img src ="examples/abalone/plots/nmse.png" />
<img src ="examples/abalone/plots/mnlp.png" />
<img src ="examples/abalone/plots/time.png" />
<h2 align="left">
Kin8nm
</h2>
<img src ="examples/kin8nm/plots/mae.png" />
<img src ="examples/kin8nm/plots/mse.png" />
<img src ="examples/kin8nm/plots/nmse.png" />
<img src ="examples/kin8nm/plots/mnlp.png" />
<img src ="examples/kin8nm/plots/time.png" />

# License
Copyright (c) 2016, Max W. Y. Lam
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.