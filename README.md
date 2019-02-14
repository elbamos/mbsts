# MBSTS

The goal of this project is to build, test, and evaluate a generalized model for Multivariate Bayesian Structural Time Series with GARCH in Stan.

The variant of this model also applies "Finnish Horseshoe" hierarchical shrinkage parameters for feature selection, including a novel application of them to seasonality and cyclicality.

## Overview

Bayesian Structural Time Series are the state-of-the-art approach to modelling time series, particularly financial time series. 

BSTS models have recently been extended to the multivariate case. See https://www.groundai.com/project/multivariate-bayesian-structural-time-series-model/ for an excellent discussion. 

BSTS models take into account local trends, seasonality, cyclicality, and external predictive variables.

This model uses "Finnish Horseshoe" hierarchical shrinkage priors  (Piironen and Vehtari 2017a) to encourage sparsity.  See https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html.

In addition, this model incorporates CCC-GARCH for handling correlated volatility shocks. 

Full list of features:
  - Linear trends (AR)
  - Seasonality, with multiple seasonality periods
  - Cyclicality
  - External predictors
  - Correlations in expected returns
  - GARCH(p,q)
  - CCC-GARCH
  - Hierarchical shrinkage priors 

## Notes on Running Stan

Models with hierarchical shrinkage priors tend to require smaller stepsizes and higher treedepths. 

If you see significant numbers of divergent transitions, consider running with 

```
samples <- sampling(model, control=list(adapt_delta=0.99, max_treedepth=15))
```

