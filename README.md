# Poisson Regression

This repository contains a Jupyter Notebook implementing Poisson regression and gradient descent optimization for modeling the number of shoppers entering a store during the day.

## Contents
- **Mathematical Formulation**: Defines the Poisson distribution, likelihood, and gradient descent update rule.
- **Implementation**: Simulates data, trains the model, and visualizes results.

## Formulation of Probabilistic Model
1. **Poisson PMF**:

   $$P(d|t, \lambda) = \frac{\lambda(t; \mathbf{w})^d \cdot \exp(-\lambda(t; \mathbf{w}))}{d!}$$

3. **Negative Log-Likelihood**:

   $$\ell(\lambda; D) = \sum_{i=1}^N \left( \lambda(t^{(i)}; \mathbf{w}) - d^{(i)} \cdot \log \lambda(t^{(i)}; \mathbf{w}) + \log d^{(i)}! \right)$$

5. **Gradient Descent Update Rule**:
   
  $$\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \frac{\partial \ell(\lambda; D)}{\partial \mathbf{w}}$$

## Dependencies
1. Install with pip: `pip install numpy matplotlib`
