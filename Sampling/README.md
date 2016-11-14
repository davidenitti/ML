# Sampling tests

##Adaptive importance sampling

The main issue of Monte-Carlo Methods, and in particular, of Importance sampling is
its high-variance in high-dimensional models with unlikely evidence.

In this project I test adaptive importance sampling schemes to solve this issue.

In adaptive importance sampling the proposal distribution used to sample is adapted during sampling.
Ideally, the proposal converges to the optimal proposal. 
If the proposal is exactly the optimal proposal, the variance of the Monte-Carlo
estimate is zero, i.e. a single sample is sufficient to obtain the exact value.

##kernel.py

This script uses the samples to approximate the posterior distribution (optimal proposal distribution).
This resemble a non-parametric kernel density estimation. 
The kernel used is the Gaussian distribution with fixed variance.


###Requirements

```
scipy
numpy
matplotlib
```  

###How to run

Execute:

```python kernel.py```

