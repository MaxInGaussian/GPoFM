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

# Performance of GPoFM on Benchmark Datasets
| Benchmark Dataset | Number of Attributes | Size of Training Data | Size of Testing Data |
| :---: | :---: | :---: | :---: |
| Bostion Housing | 13 | 400 | 106 |
| Abalone | 10 | 3133 | 1044 |
| Kin8nm | 10 | 5000 | 3192 |

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

#License
Copyright (c) 2016, Max W. Y. Lam
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.