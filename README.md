# MBSTS

The goal of this project is to build, test, and evaluate a generalized model for Multivariate Bayesian Structural Time Series in Stan.

## Overview

Bayesian Structural Time Series are the state-of-the-art approach to modelling time series, particularly financial time series. 

BSTS models take into account local trends, seasonality, cyclicality, and external predictive variables. This is an ideal problem for Bayesian inference, for reasons I will explain when I have time.

Currently the model includes the following features for multivariate time series:
  - Linear trends (AR)
  - Seasonality, where each price series may be given its own seasonal period
  - Cyclicality
  - External predictors
  - Correlations in expected returns
  - GARCH
  - CCC-GARCH