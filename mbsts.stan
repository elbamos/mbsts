/*
Potential issues:
---- Hierarchical shrinkage ----
- How do we choose the slab_scale? currently using 1
- Which volatility to use in calculating tau0? currently using sigma_price

---- Cyclicality ---- 
I'm not 100 % confident that I'm correctly interpreting the formula

---- Other ---- 
- Do we want shrinkage priors on the trend, seasonality, or price shocks themselves?
  - Perhaps the shrinkage prior could be used for theta on seasonality; the theory being, if a price series isn't seasonal, the parameter will shrink toward zero.
- Prior on seasonality? 
- Prior on lambda for cyclicality? 
- Prior on the damping factor for cyclicality?

--- CHANGES ---
- Gave in and zero-centered the draws of the pre-shrunk params
*/

functions {
  
  matrix make_L(row_vector theta, matrix Omega) {
    return diag_pre_multiply(sqrt(theta), Omega);
  }
  
  // Linear Trend 
  row_vector make_delta_t(row_vector alpha_trend, matrix beta_trend, matrix delta_past, row_vector nu) {
      return alpha_trend + columns_dot_product(beta_trend, delta_past - rep_matrix(alpha_trend, rows(delta_past))) + nu;
  }
  
  // Sparsity
  void hs_prior_lp(vector param, real tau, vector lambda_m, real c2_tilde, real nu) {
    tau ~ cauchy(0, 1);
    c2_tilde ~ inv_gamma(nu/2, nu / 2);
    lambda_m ~ cauchy(0, 1);
    param ~ normal(0, 1);
  }

  matrix apply_hs_prior(matrix param_raw, real target_sparsity, int N, real sigma, real slab_scale, real tau, vector lambda_m, real c2_tilde) {
    int M = num_elements(param_raw);
    real m0 =  M * target_sparsity;
    real tau0 = (m0 / (M - m0)) * (sigma / sqrt(1.0 * N));
    vector[M] lambda_m_tilde;
    real c2 = c2_tilde * slab_scale;
    lambda_m_tilde = sqrt(c2 * square(lambda_m) ./ (c2 + square(tau * tau0) * square(lambda_m)));
    return tau * tau0 * to_matrix(lambda_m_tilde, rows(param_raw), cols(param_raw)) .* param_raw;
  }
}

data { 
  int<lower=2> N; // Number of price points
  int<lower=2> N_series; // Number of price series
  int<lower=2> N_periods; // Number of periods
  int<lower=1> N_features; // Number of features in the regression
  
  // Parameters controlling the time series 
  int<lower=2> periods_to_predict;
  int<lower=1> ar; // AR period for the trend
  int<lower=1> p; // GARCH
  int<lower=1> q; // GARCH
  int<lower=1> s[N_series]; // seasonality periods
  
  // Parameeters controlling sparse feature selection
  real<lower=0,upper=1>              m0; // Sparsity target; proportion of features we expect to be relevant
  int<lower=1>                       nu; // nu parameters for horseshoe prior
  
  // Data 
  vector<lower=0>[N]                         y;
  int<lower=1,upper=N_periods>               period[N];
  int<lower=1,upper=N_series>                series[N];
  vector<lower=0>[N]                         weight;
  matrix[N_periods, N_features]              x; // Regression predictors
  
  matrix[periods_to_predict, N_features]     x_predictive;
}

transformed data {
  vector<lower=0>[N]                  log_y;
  real<lower=0>                       min_price =  log1p(min(y));
  real<lower=0>                       max_price = log1p(max(y));
  row_vector[N_series]                zero_vector = rep_row_vector(0, N_series);

  for (n in 1:N) {
    log_y[n] = log1p(y[n]);
  }
}


parameters {
  real<lower=0>                                sigma_y; // observation variance
  
  // TREND delta_t
  matrix[1, N_series]                                 delta_t0; // Trend at time 0
  row_vector[N_series]                                alpha_trend; // long-term trend
  // Hierarchical shrinkage prior on beta_trend 
  vector<lower=0>[ar * N_series]                      lambda_m_beta_trend; 
  real<lower=0>                                       c_beta_trend;
  real<lower=0>                                       tau_beta_trend;
  matrix<lower=0,upper=1>[ar, N_series]               beta_trend; // Learning rate of trend
  row_vector[N_series]                                nu_trend[N_periods-1]; // Random changes in trend
  row_vector<lower=0>[N_series]                       theta_trend; // Variance in changes in trend
  cholesky_factor_corr[N_series]                      L_omega_trend; // Correlations among trend changes
  
  // SEASONALITY
  row_vector[N_series]                                w_t[N_periods-1]; // Random variation in seasonality
  vector<lower=0>[N_series]                           theta_season; // Variance in seasonality

  // CYCLICALITY
  row_vector<lower=0, upper=pi()>[N_series]           lambda; // Frequency
  row_vector<lower=0, upper=1>[N_series]              rho; // Damping factor
  vector<lower=0>[N_series]                           theta_cycle; // Variance in cyclicality
  matrix[N_periods - 1, N_series]                     kappa;  // Random changes in cyclicality
  matrix[N_periods - 1, N_series]                     kappa_star; // Random changes in counter-cyclicality
  
  // REGRESSION
  vector<lower=0>[N_features * N_series]              lambda_m_beta_xi; 
  real<lower=0>                                       c_beta_xi;
  real<lower=0>                                       tau_beta_xi;
  matrix[N_features, N_series]                        beta_xi; // Coefficients of the regression parameters
  
  // INNOVATIONS
  matrix[N_periods-1, N_series]                       epsilon; // Innovations
  row_vector<lower=0>[N_series]                       omega_garch; // Baseline volatility of innovations
    // Hierarchical shrinkage prior on beta_p
  vector<lower=0>[p * N_series]                       lambda_m_beta_p; 
  real<lower=0>                                       c_beta_p;
  real<lower=0>                                       tau_beta_p;
  matrix<lower=0>[p, N_series]                        beta_p; // Univariate GARCH coefficients on prior volatility
  // Hierarchical shrinkage prior on beta_q
  vector<lower=0>[q * N_series]                       lambda_m_beta_q; 
  real<lower=0>                                       c_beta_q;
  real<lower=0>                                       tau_beta_q;
  matrix<lower=0>[q, N_series]                        beta_q; // Univariate GARCH coefficients on prior innovations
  cholesky_factor_corr[N_series]                      L_omega_garch; // Constant correlations among innovations 
  
  row_vector<lower=min_price,upper=max_price>[N_series] starting_prices;
}

transformed parameters {
  matrix[N_periods, N_series]                         log_prices_hat; // Observable prices
  matrix[N_periods-1, N_series]                       delta; // Trend at time t
  matrix[N_periods-1, N_series]                       tau; // Seasonality at time t
  matrix[N_periods-1, N_series]                       omega; // Cyclicality at time t
  matrix[N_periods-1, N_series]                       omega_star; // Anti-cyclicality at time t
  matrix[N_periods-1, N_series]                       theta; // Conditional variance of innovations 
  matrix[N_periods, N_series]                         xi = x * beta_xi; // Predictors
  vector[N]                                           log_y_hat; 
  matrix[N_series, N_series]                          L_Omega_trend = make_L(theta_trend, L_omega_trend);
  row_vector[N_series] rho_cos_lambda = rho .* cos(lambda); 
  row_vector[N_series] rho_sin_lambda = rho .* sin(lambda); 
    // TODO: Is sigma_y the correct volatility to use here?
  // TODO: How do we determine the slab scale? 
  matrix[ar, N_series] beta_trend_hs = apply_hs_prior(beta_trend, m0, N_periods, sigma_y, 1, tau_beta_trend, lambda_m_beta_trend, c_beta_trend); 
  matrix[ar, N_series] beta_p_hs = apply_hs_prior(beta_p, m0, N_periods, sigma_y, 1, tau_beta_p, lambda_m_beta_p, c_beta_p); 
  matrix[ar, N_series] beta_q_hs = apply_hs_prior(beta_q, m0, N_periods, sigma_y, 1, tau_beta_q, lambda_m_beta_q, c_beta_q); 
  
  // TREND
  delta[1] = make_delta_t(alpha_trend, block(beta_trend_hs, ar, 1, 1, N_series), delta_t0, nu_trend[1]);
  for (t in 2:(N_periods-1)) {
    if (t <= ar) {
      delta[t] = make_delta_t(alpha_trend, block(beta_trend_hs, ar - t + 2, 1, t - 1, N_series), block(delta, 1, 1, t - 1, N_series), nu_trend[t]);
    } else {
      delta[t] = make_delta_t(alpha_trend, beta_trend_hs, block(delta, t - ar, 1, ar, N_series), nu_trend[t]);
    }
  }

  // ----- SEASONALITY ------
  tau[1] = -w_t[1];
  for (t in 2:(N_periods-1)) {
    for (d in 1:N_series) {
      tau[t, d] = -sum(sub_col(tau, max(1, t - 1 - s[d] - 1), d, min(s[d] - 1, t - 1)));
    }
    tau[t] += w_t[t];
  }
  
  // ----- CYCLICALITY ------
  omega[1] = kappa[1];
  omega_star[1] = kappa_star[1]; 
  for (t in 2:(N_periods-1)) {
    omega[t] = (rho_cos_lambda .* omega[t - 1]) + (rho_sin_lambda .* omega_star[t-1]) + kappa[t];
    # TODO: Confirm that the negative only applies to the first factor not both
    omega_star[t] = - (rho_sin_lambda .* omega[t - 1]) + (rho_cos_lambda .* omega_star[t-1]) + kappa_star[t];
  }
  
  // ----- UNIVARIATE GARCH ------
  theta[1] = omega_garch; 
  {
    matrix[N_periods-1, N_series] epsilon_squared = square(epsilon);
    
    for (t in 2:(N_periods-1)) {
      row_vector[N_series]  p_component; 
      row_vector[N_series]  q_component; 
      
      if (t <= p) {
        p_component = columns_dot_product(block(beta_p_hs, p - t + 2, 1, t - 1, N_series), block(theta, 1, 1, t - 1, N_series));
      } else {
        p_component = columns_dot_product(beta_p_hs, block(theta, t - p, 1, p, N_series));
      }
      
      if (t <= q) {
        q_component = columns_dot_product(block(beta_q_hs, q - t + 2, 1, t - 1, N_series), block(epsilon_squared, 1, 1, t - 1, N_series));
      } else {
        q_component = columns_dot_product(beta_q_hs, block(epsilon_squared, t - q, 1, q, N_series));
      }
      
      theta[t] = omega_garch + p_component + q_component;
    }
  }

  // ----- ASSEMBLE TIME SERIES ------

  log_prices_hat[1] = starting_prices + xi[1]; 
  for (t in 2:N_periods) {
    log_prices_hat[t] = log_prices_hat[t-1] + delta[t-1] + tau[t-1] + omega[t-1] + xi[t-1] + epsilon[t-1];
  }
  
  for (n in 1:N) {
    log_y_hat[n] = log_prices_hat[period[n], series[n]];
  }
}


model {
  vector[N] price_error = log_y - log_y_hat;
  
  // ----- PRIORS ------
  // TREND 
  to_vector(delta_t0) ~ normal(0, 1); 
  to_vector(alpha_trend) ~ normal(0, 1); 
  hs_prior_lp(to_vector(beta_trend), tau_beta_trend, lambda_m_beta_trend, c_beta_trend, nu);
  to_vector(theta_trend) ~ cauchy(0, 1); 
  L_omega_trend ~ lkj_corr_cholesky(1);

  // SEASONALITY
  theta_season ~ cauchy(0, 1); 

  // CYCLICALITY
  lambda ~ uniform(0, pi());
  rho ~ uniform(0, 1);
  theta_cycle ~ cauchy(0, 1);

  // REGRESSION
  hs_prior_lp(to_vector(beta_xi), tau_beta_xi, lambda_m_beta_xi, c_beta_xi, nu);

  // INNOVATIONS
  omega_garch ~ cauchy(0, 1);
  hs_prior_lp(to_vector(beta_p), tau_beta_p, lambda_m_beta_p, c_beta_p, nu);
  hs_prior_lp(to_vector(beta_q), tau_beta_q, lambda_m_beta_q, c_beta_q, nu);
  L_omega_garch ~ lkj_corr_cholesky(1);

  // ----- TIME SERIES ------
  // Time series
  to_vector(starting_prices) ~ uniform(min_price, max_price); 
  nu_trend ~ multi_normal_cholesky(zero_vector, L_Omega_trend);
  for (t in 1:(N_periods-1)) {
    w_t[t] ~ normal(zero_vector, theta_season);
    kappa[t] ~ normal(zero_vector, theta_cycle);
    kappa_star[t] ~ normal(zero_vector, theta_cycle);
    epsilon[t] ~ multi_normal_cholesky(zero_vector, make_L(theta[t], L_omega_garch));
  }


  // ----- OBSERVATIONS ------
  sigma_y ~ cauchy(0, 0.01);
  price_error ~ normal(0, inv(weight) * sigma_y);
}

generated quantities {
  matrix[periods_to_predict, N_series]             log_predicted_prices; 
  matrix[periods_to_predict, N_series]             delta_hat; // Trend at time t
  matrix[periods_to_predict, N_series]             tau_hat; // Seasonality at time t
  matrix[periods_to_predict, N_series]             omega_hat; // Cyclicality at time t
  matrix[periods_to_predict, N_series]             omega_star_hat; // Anti-cyclicality at time t
  matrix[periods_to_predict, N_series]             theta_hat; // Conditional variance of innovations 
  matrix[periods_to_predict, N_series]             epsilon_hat; 
  matrix[periods_to_predict, N_series]             xi_hat = x_predictive * beta_xi;
  matrix[periods_to_predict, N_series]             nu_trend_hat; 
  matrix[periods_to_predict, N_series]             kappa_hat;
  matrix[periods_to_predict, N_series]             kappa_star_hat; 
  matrix[periods_to_predict, N_series]             w_t_hat;
  
  for (t in 1:periods_to_predict) {
    nu_trend_hat[t] = multi_normal_cholesky_rng(to_vector(zero_vector), L_Omega_trend)';
    kappa_hat[t] = multi_normal_rng(zero_vector', diag_matrix(theta_cycle))';
    kappa_star_hat[t] = multi_normal_rng(zero_vector', diag_matrix(theta_cycle))';
    w_t_hat[t] = multi_normal_rng(zero_vector', diag_matrix(theta_season))';
  }
  
  // TREND
  for (t in 1:periods_to_predict) {
    if (t == 1) {
      delta_hat[1] = make_delta_t(alpha_trend, beta_trend_hs, block(delta, N_periods - ar, 1, ar, N_series), nu_trend_hat[1]);
    } else if (t <= ar) {
      int periods_forward = t - 1;
      int periods_back = ar - periods_forward;
      int start_period = N_periods- 1 - periods_back; 
      delta_hat[t] = make_delta_t(alpha_trend, beta_trend_hs, append_row(block(delta, start_period, 1, periods_back, N_series), 
                                                                      block(delta_hat, 1, 1, periods_forward, N_series)), nu_trend_hat[t]);
    } else {
      delta_hat[t] = make_delta_t(alpha_trend, beta_trend_hs, block(delta_hat, t - ar, 1, ar, N_series), nu_trend_hat[t]);
    }
  }
  
  // SEASONALITY
  for (t in 1:(periods_to_predict)) {
    for (d in 1:N_series) {
      if (t < s[d]) {
        tau_hat[t, d] = -sum(append_row(
          sub_col(tau_hat, 1, d, t - 1), 
          sub_col(tau, N_periods - 1 - (s[d] - t), d, s[d] - t)
        )) + w_t_hat[t, d];
      } else {
        tau_hat[t, d] = -sum(sub_col(tau_hat, t - s[d] + 1, d, s[d] - 1)) + w_t_hat[t, d];
      }
    }
  }
  
  // Cyclicality
  for (t in 1:(periods_to_predict)) {
    if (t == 1) {
      omega_hat[t] = (rho_cos_lambda .* omega[N_periods-1]) + (rho_sin_lambda .* omega_star[N_periods-1]) + kappa_hat[t];
      omega_star_hat[t] = -(rho_sin_lambda .* omega[N_periods-1]) + (rho_cos_lambda .* omega_star[N_periods-1]) + kappa_star_hat[t];
    } else {
      omega_hat[t] = (rho_cos_lambda .* omega_hat[t-1]) + (rho_sin_lambda .* omega_star_hat[t-1]) + kappa_hat[t];
      omega_star_hat[t] = -(rho_sin_lambda .* omega_hat[t-1]) + (rho_cos_lambda .* omega_star_hat[t-1]) + kappa_star_hat[t];   
    }
  }
  
  
  // Univariate GARCH
  for (t in 1:periods_to_predict) {
    row_vector[N_series]  p_component; 
    row_vector[N_series]  q_component; 
    
    if (t == 1) {
      p_component = columns_dot_product(beta_p_hs, block(theta, N_periods - 1 - p, 1, p, N_series));
    } else if (t <= p) {
      int periods_forward = t - 1;
      int periods_back = p - periods_forward;
      int start_period = N_periods- 1 - periods_back; 
      p_component = columns_dot_product(beta_p_hs, append_row(
        block(theta, start_period, 1, periods_back, N_series),
        block(theta_hat, 1, 1, periods_forward, N_series)
      ));
    } else {
      p_component = columns_dot_product(beta_p_hs, block(theta_hat, t - p, 1, p, N_series));
    }
    
    if (t == 1) {
      q_component = columns_dot_product(beta_q_hs, square(block(epsilon, N_periods - 1 - q, 1, q, N_series)));
    } else if (t <= q) {
      int periods_forward = t - 1;
      int periods_back = q - periods_forward;
      int start_period = N_periods- 1 - periods_back; 
      q_component = columns_dot_product(beta_q, square(append_row(
        block(epsilon, start_period, 1, periods_back, N_series),
        block(epsilon_hat, 1, 1, periods_forward, N_series)
      ))); 
    } else {
      q_component = columns_dot_product(beta_q_hs, square(block(epsilon_hat, t - q, 1, q, N_series)));
    }
    
    theta_hat[t] = omega_garch + p_component + q_component;
    epsilon_hat[t] = multi_normal_cholesky_rng(zero_vector', make_L(theta_hat[t], L_omega_garch))';
  }
  
  
  log_predicted_prices[1] = log_prices_hat[N_periods] + delta_hat[1] + tau_hat[1] + omega_hat[1] + xi_hat[1] + epsilon_hat[1];
  for (t in 2:periods_to_predict) {
    log_predicted_prices[t] = log_predicted_prices[t-1] + delta_hat[t] + tau_hat[t] + omega_hat[t] + xi_hat[t] + epsilon_hat[t];
  }
}

