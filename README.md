#GPoFM

GPoFM is a machine learning toolkit designed for predictive modeling of
complex-structured data. The key of success is to find a tailormade function
that maps the inputs to the targets. In this sense, linear functions undoubtedly
would be too simple to solve the problem. Nevertheless, non-linear functions
with much flexibility are mostly concerned. In fact, there are uncountable ways
to define an non-linear function, and it's generally hard to tell which class of
mathematical functions specifically works for a problem. A machine-learning kind
of approach is to approximate such a function by 'supervising' data and
'learning' patterns. In statistical community, this is traditionally coined
'regression'. Although such a data-driven function can be obtained through
optimization, the optimized models tend to lose generality, which is technically
regarded as 'overfitting'.

To prevent from overfitting, one feasible approach is to carry out Bayesian
inference over the distribution of non-linear functions. That is, the desired
function is assumed to be 'produced' from a particular space. Due to
mathematical brevity and elegance, Gaussian process is mainly employed to
describe the distribution over functions. [Carl Edward Rasmussen and Christopher
K. I. Williams](http://www.gaussianprocess.org/gpml/), who pioneer and
popularize the idea of using Gaussian processes for machine learning tasks,
emphasize that one of the greatest advantages of Gaussian process is that we can
integrate all possible functions over the function distribution (Gaussian
process), and obtain an analytical solution because of nice properties of
Gaussian. It is pinpointed that this Bayesian routine is prefered over
optimization on a certain estimate of function.

In GPoFM, the 
proposed improvement of [Sparse Spectrum Gaussian Process](http://quinonero.net/Publications/lazaro-gredilla10a.pdf) (SSGP), which is a new branch of method to speed up Gaussian process model taking advantage of [Fourier features](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf). Recall that using 

Based on minimization of the marginal likelihood, SCFGP selects a set of vectors to obtain a [Gramian matrix](https://en.wikipedia.org/wiki/Gramian_matrix), which is treated as the frequency matrix for later computation of Fourier features. This procedure indeed can be viewed as constructing sparsely correlated Fourier features. 

Note that the Fourier features are identically and independently distributed in SSGP, therefore the size of optimization parameters is proportional to the number of Fourier features times the number of dimension. This is undoubtedly an unfavorable property, since the model is likely to stick in local minima and becomes very unstable when dealing with very high dimensional data, such as images, speech signals, text, etc.

The formulation of SCFGP is briefly described in this sheet: (Derivation will be included in the future)

![SCFGP Formulas](SCFGP_formulas.png?raw=true "SCFGP Formulas")

GPoFM is developed in python using Theano and originally designed by
Max W. Y. Lam (maxingaussian@gmail.com).

###Highlights of SCFGP

- SCFGP optimizes
the Fourier features so as to "learn" a tailmade covariance matrix from the data. 
This removes the necessity of deciding which kernel function to use in different problems.

- SCFGP implements a variety of formulation to transform the optimized Fourier features to covariance matrix, including the typical sin-cos concatenation introduced by [Miguel](http://www.jmlr.org/papers/v11/lazaro-gredilla10a.html), and the generalized approach described by [Yarin](http://jmlr.org/proceedings/papers/v37/galb15.html).

- SCFGP uses low-rank frequency matrix for sparse approximation of Fourier features. It is 
intended to show that low-rank frequency matrix is able to lower the computational 
burden in each step of optimization, and also render faster convergence and a stabler result.

- Compared with other 
state-of-the-art regressors, SCFGP usually gives the most accurate prediction on the benchmark datasets of regression.

# Installation
   
### SCFGP

To install SCFGP, use pip:

    $ pip install SCFGP

Or clone this repo:

    $ git clone https://github.com/MaxInGaussian/SCFGP.git
    $ python setup.py install

## Dependencies
### Theano
    Theano is used due to its nice and simple syntax to set up the tedious formulas in SCFGP, and
    its capability of computing automatic differentiation.
    
To install Theano, see this page:

   http://deeplearning.net/software/theano/install.html

### scikit-learn (only used in the examples)
    
To install scikit-learn, see this page:

   https://github.com/scikit-learn/scikit-learn
# Try SCFGP with Only 3 Lines of Code
```python
from SCFGP import *
# <>: necessary inputs, {}: optional inputs
model = SCFGP(rank=<rank_of_frequency_matrix>,
              feature_size=<number_of_Fourier_features>,
              fftype={feature_type},
              msg={print_message_or_not})
model.fit(X_train, y_train, {X_test}, {y_test})
predict_mean, predict_std = model.predict(X_test, {y_test})
```

# Analyze Training Process on Real Time
## Training on High-dimensional Inputs (Boston Housing)
```python
model.fit(X_train, y_train, X_test, y_test, plot_training=True)
```
![PlotTraining](examples/plot_training.gif?raw=true "Plot Training")
## Training on One-dimensional Inputs (Mauna Loa Atmospheric CO2)
```python
model.fit(X_train, y_train, X_test, y_test, plot_1d_function=True)
```
![Plot1DFunction](examples/plot_1d_function.gif?raw=true "Plot 1D Function")

# Performance of SCFGP on Benchmark Datasets
| Benchmark Dataset | Number of Attributes | Size of Training Data | Size of Testing Data |
| :---: | :---: | :---: | :---: |
| Bostion Housing | 13 | 400 | 106 |
| Abalone | 10 | 3133 | 1044 |
| Kin8nm | 10 | 5000 | 3192 |
##Predict Boston Housing Prices
| Regression Model | MAE | MSE | RMSE | NMSE | MNLP | Training Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SCFGP** | **1.3398** | **3.1828** | **1.7841** | **0.0405** | **2.0106** | 12.8740 |
| [Boosting](http://professordrucker.com/Pubications/ImprovingRegressorsUsingBoostingTechniques.pdf) | N/A | 10.7 | N/A | N/A | N/A | N/A |
| [MARK-L](http://dl.acm.org/citation.cfm?id=775051) | N/A | 12.4 | N/A | N/A | N/A | **6.0** |
| [PS-SVR](http://epubs.siam.org/doi/abs/10.1137/1.9781611972726.16) | N/A | 7.887 | N/A | 0.0833 | N/A | N/A |
| [spLGP](http://www2.stat.duke.edu/~st118/Publication/TokdarZhuGhosh.pdf) | 1.73 | N/A | N/A | N/A | N/A | N/A |
| [Student-t GP](http://people.ee.duke.edu/~lcarin/NIPS2009_0224.pdf) | N/A | N/A | N/A | 0.0824 | N/A | N/A |

P.S. SCFGP's performance refers to this model:
```python
boston_housing_best_model = SCFGP()
boston_housing_best_model.load("examples/boston_housing/best_model.pkl")
```

## Predict Age of Abalone
| State-Of-The-Art Model | MAE | MSE | RMSE | NMSE | MNLP | Training Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SCFGP** | **1.4113** | **3.8153** | **1.9533** | **0.3715** | **2.0916** | **9.5621** |
| [MARK-L](http://dl.acm.org/citation.cfm?id=775051) | N/A | 4.65 | N/A | N/A | N/A | 57.0 |

P.S. SCFGP's performance refers to this model:
```python
abalone_best_model = SCFGP()
abalone_best_model.load("examples/abalone/best_model.pkl")
```

## Predict Kinematics of 8-link Robot Arm
| State-Of-The-Art Model | MAE | MSE | RMSE | NMSE | MNLP | Training Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SCFGP** | **0.0561** | **0.0052** | **0.0718** | **0.0741** | **-1.2170** | **355.6762** |


P.S. SCFGP's performance refers to this model:
```python
kin8nm_best_model = SCFGP()
kin8nm_best_model.load("examples/kin8nm/best_model.pkl")
```

# Performance of SCFGP v.s. Number of Fourier features
<h2 align="center">
Bostion Housing
</h2>
![BostonHousingMAE](examples/boston_housing/plots/mae.png?raw=true "Boston Housing MAE")
![BostonHousingMSE](examples/boston_housing/plots/mse.png?raw=true "Boston Housing MSE")
![BostonHousingRMSE](examples/boston_housing/plots/rmse.png?raw=true "Boston Housing RMAE")
![BostonHousingNMSE](examples/boston_housing/plots/nmse.png?raw=true "Boston Housing NMSE")
![BostonHousingMNLP](examples/boston_housing/plots/mnlp.png?raw=true "Boston Housing MNLP")
![BostonHousingTime](examples/boston_housing/plots/time(s).png?raw=true "Boston Housing Time")
<h2 align="center">
Abalone
</h2>
![AbaloneMAE](examples/abalone/plots/mae.png?raw=true "Abalone MAE")
![AbaloneMSE](examples/abalone/plots/mse.png?raw=true "Abalone MSE")
![AbaloneRMSE](examples/abalone/plots/rmse.png?raw=true "Abalone RMAE")
![AbaloneNMSE](examples/abalone/plots/nmse.png?raw=true "Abalone NMSE")
![AbaloneMNLP](examples/abalone/plots/mnlp.png?raw=true "Abalone MNLP")
![AbaloneTime](examples/abalone/plots/time(s).png?raw=true "Abalone Time")
<h2 align="center">
Kin8nm
</h2>
![Kin8nmMAE](examples/kin8nm/plots/mae.png?raw=true "Kin8nm MAE")
![Kin8nmMSE](examples/kin8nm/plots/mse.png?raw=true "Kin8nm MSE")
![Kin8nmRMSE](examples/kin8nm/plots/rmse.png?raw=true "Kin8nm RMAE")
![Kin8nmNMSE](examples/kin8nm/plots/nmse.png?raw=true "Kin8nm NMSE")
![Kin8nmMNLP](examples/kin8nm/plots/mnlp.png?raw=true "Kin8nm MNLP")
![Kin8nmTime](examples/kin8nm/plots/time(s).png?raw=true "Kin8nm Time")
<h3 align="center">
Training time of SCFGP is not quite sensitive to the size of training data.<br>
On the contrary, it is to a large extent dependent on the number of Fourier features.
</h3>

#License
Copyright (c) 2016, Max W. Y. Lam
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.