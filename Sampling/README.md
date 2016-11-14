# Sampling tests

##Adaptive importance sampling

The main issue of Monte-Carlo Methods, and in particular, of Importance sampling is
its high-variance in high-dimensional models with unlikely evidence.

In this project I test adaptive importance sampling schemes to solve this issue.

In adaptive importance sampling the proposal distribution used to sample is adapted during sampling.
Ideally, the proposal converges to the optimal proposal. 
If the proposal is exactly the optimal proposal, the variance of the Monte-Carlo
estimate is zero, i.e. a single sample is sufficient to obtain the exact value.

The task is to estimate the integral of a function f(x).
The optimal proposal distribution is f(x)/M, where M is the integral of f(x).
Thus the optimal proposal distribution is f(x) normalized.

##kernel.py

This script uses the samples to approximate the optimal proposal distribution.
The optimal proposal distribution is approximated as a mixture of Gaussians centered in each sample.
This resemble a non-parametric kernel density estimation. 
The kernel used is the Gaussian distribution with fixed variance.


##nnleastsquare.py

This script uses Gaussian kernels as the previous example, but the proposal distribution is estimated fitting
the density of a mixture of Gaussians with optimal proposal estimates f(x_i)/M for each sample x_i.
Obviously M is not known, thus it is approximated using importance sampling.
The density is fitted using a non-negative least squares, that is the weights of the regression task must be non-negative.
Note that this is different from standard maximum-likelihood estimations.


###Requirements

```
scipy
numpy
matplotlib
```  

###How to run

Execute:

```python kernel.py```

```python nnleastsquare.py```
