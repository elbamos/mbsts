/*
Potential issues:
---- Hierarchical shrinkage ----
- How do we choose the slab_scale? currently using 1
- Which volatility to use in calculating tau0? currently using sigma_price

---- Other ---- 
- Do we want shrinkage priors on the trend, seasonality, or price shocks themselves?
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
  row_vector make_delta_t_ar1(row_vector alpha_trend, row_vector beta_trend, row_vector delta_past, row_vector nu) {
      return alpha_trend + (beta_trend .* (delta_past - alpha_trend)) + nu;
  }
  
  row_vector make_delta_t(row_vector alpha_trend, matrix beta_trend, matrix delta_past, row_vector nu) {
      return alpha_trend + columns_dot_product(beta_trend, delta_past) + nu;
  }
  
  // Sparsity
  void hs_prior_lp(vector param, real tau, vector lambda_m, real c2_tilde, real slab_scale, real nu) {
    tau ~ cauchy(0, 1);
    c2_tilde ~ inv_gamma(nu/2.0, square(slab_scale) * nu / 2.0);
    lambda_m ~ cauchy(0, 1);
    param ~ normal(0, 1);
  }

  real get_tau0_start(real target_sparsity, real M, int N) {
    real m0 =  M * target_sparsity;
    // This should be sigma / sqrt(N). But, 
    // the remainder is constant, so we calculate it once and multiply by sigma 
    // during inference.
    return (m0 / (M - m0)) * (1 / sqrt(1.0 * N));
  }
  
  matrix apply_hs_prior_m(matrix param_raw, real tau0_start, real sigma, real tau, vector lambda_m, real c2) {
    int M = num_elements(param_raw);
    real tau0tau = tau0_start * sigma * tau;
    vector[M] lambda_m_square = square(lambda_m);
    vector[M] lambda_m_tilde = sqrt(c2 * lambda_m_square ./ (c2 + square(tau0tau) * lambda_m_square));
    return tau0tau * to_matrix(lambda_m_tilde, rows(param_raw), cols(param_raw)) .* param_raw;
  }
  
  vector apply_hs_prior_v(vector param_raw, real tau0_start, real sigma, real tau, vector lambda_m, real c2) {
    int M = num_elements(param_raw);
    real tau0tau = tau0_start * sigma * tau;
    vector[M] lambda_m_square = square(lambda_m);
    vector[M] lambda_m_tilde = sqrt(c2 * lambda_m_square ./ (c2 + square(tau0tau) * lambda_m_square));
    return tau0tau * lambda_m_tilde .* param_raw;
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
  int<lower=1> N_seasonality;
  int<lower=1> s[N_seasonality]; // seasonality 
  // A prior on the scale of each factor; a reasonable estimate the number of periods over which a price could reasonably double
  real<lower=1> period_scale; 
  
  // Parameeters controlling sparse feature selection
  real<lower=0,upper=1>              m0; // Sparsity target; proportion of features we expect to be relevant
  int<lower=1>                       nu; // nu parameters for horseshoe prior

  // Data 
  vector<lower=0>[N]                         y;
  int<lower=1,upper=N_periods>               period[N];
  int<lower=1,upper=N_series>                series[N];
  vector<lower=0>[N]                         weight;
  matrix[N_periods, N_features]              x; // Regression predictors
}

transformed data {
  vector<lower=0>[N]                  log_y;
  real<lower=0>                       min_price =  log1p(min(y));
  real<lower=0>                       max_price = log1p(max(y));
  row_vector[N_series]                zero_vector = rep_row_vector(0, N_series);
  vector<lower=0>[N]                  inv_weights;
  real<lower=0>                       inv_period_scale = 1.0 / period_scale; 
  real                                tau0_vector = get_tau0_start(m0, N_series, N_periods);
  real                                tau0_ar = get_tau0_start(m0, N_series * ar, N_periods);
  real                                tau0_beta_p = get_tau0_start(m0, N_series * p, N_periods);
  real                                tau0_beta_q = get_tau0_start(m0, N_series * q, N_periods);
  real                                tau0_xi = get_tau0_start(m0, N_series * N_features, N_periods);
  real                                tau0_theta_cycle = get_tau0_start(m0, N_series, N_periods);
  real                                tau0_theta_season = get_tau0_start(m0, N_series * N_seasonality, N_periods);
  real                                min_beta_ar;

  if (ar == 1) min_beta_ar = 0;
  else min_beta_ar = -1;

  for (n in 1:N) {
    log_y[n] = log1p(y[n]);
    inv_weights[n] = 1.0 / weight[n];
  }
}


parameters {
  real<lower=0>                                       sigma_y; // observation variance
  
  // TREND delta_t
  matrix[1, N_series]                                 delta_t0; // Trend at time 0
  row_vector[N_series]                                alpha_ar; // long-term trend
  // Hierarchical shrinkage prior on beta_trend 
  vector<lower=0>[ar * N_series]                      lambda_m_beta_ar; 
  real<lower=0>                                       c_beta_ar;
  real<lower=0>                                       tau_beta_ar;
  matrix<lower=0,upper=1>[ar, N_series]               beta_ar; // Learning rate of trend
  row_vector[N_series]                                nu_trend[N_periods-1]; // Random changes in trend
  row_vector<lower=0>[N_series]                       theta_ar; // Variance in changes in trend
  cholesky_factor_corr[N_series]                      L_omega_ar; // Correlations among trend changes
  
  // SEASONALITY
  row_vector[N_series]                                w_t[N_seasonality, N_periods-1]; // Random variation in seasonality
  vector<lower=0>[N_series]                           lambda_m_theta_season[N_seasonality]; 
  real<lower=0>                                       c_theta_season;
  real<lower=0>                                       tau_theta_season;
  vector<lower=0>[N_series]                           theta_season[N_seasonality]; // Variance in seasonality

  // CYCLICALITY
  row_vector<lower=0, upper=pi()>[N_series]           lambda; // Frequency
  row_vector<lower=0, upper=1>[N_series]              rho; // Damping factor
  // Hierarchical shrinkage prior on beta_trend 
  vector<lower=0>[N_series]                           lambda_m_theta_cycle; 
  real<lower=0>                                       c_theta_cycle;
  real<lower=0>                                       tau_theta_cycle;
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
  matrix[N_periods-1, N_series]                       tau_s[N_seasonality]; // Seasonality for each periodicity
  matrix[N_periods-1, N_series]                       tau; // Total seasonality
  matrix[N_periods-1, N_series]                       omega; // Cyclicality at time t
  matrix[N_periods-1, N_series]                       omega_star; // Anti-cyclicality at time t
  matrix[N_periods-1, N_series]                       theta; // Conditional variance of innovations 
  vector[N]                                           log_y_hat; 
  matrix[N_series, N_series]                          L_Omega_ar = make_L(theta_ar, L_omega_ar);
  // Helpers for calculating cyclicality
  row_vector[N_series] rho_cos_lambda = rho .* cos(lambda); 
  row_vector[N_series] rho_sin_lambda = rho .* sin(lambda); 
  // Hierarchical shrinkage
  matrix[ar, N_series] beta_ar_hs = apply_hs_prior_m(beta_ar, tau0_ar, sigma_y, tau_beta_ar, lambda_m_beta_ar, c_beta_ar); 
  matrix[ar, N_series] beta_p_hs = apply_hs_prior_m(beta_p, tau0_beta_p, sigma_y, tau_beta_p, lambda_m_beta_p, c_beta_p); 
  matrix[ar, N_series] beta_q_hs = apply_hs_prior_m(beta_q, tau0_beta_q, sigma_y, tau_beta_q, lambda_m_beta_q, c_beta_q); 
  matrix[N_features, N_series] beta_xi_hs = apply_hs_prior_m(beta_xi, tau0_xi, sigma_y, tau_beta_xi, lambda_m_beta_xi, c_beta_xi); 
  vector[N_series] theta_cycle_hs = apply_hs_prior_v(theta_cycle, tau0_theta_cycle, sigma_y, tau_theta_cycle, lambda_m_theta_cycle, c_theta_cycle);
  vector[N_series] theta_season_hs[N_seasonality];

  // Retain calculation of xi
  matrix[N_periods, N_series]                         xi = x * beta_xi_hs; // Predictors

  // TREND
  if (ar > 1) {
    delta[1] = make_delta_t(alpha_ar, block(beta_ar_hs, ar, 1, 1, N_series), delta_t0, nu_trend[1]);
    for (t in 2:(N_periods-1)) {
      if (t <= ar) {
        delta[t] = make_delta_t(alpha_ar, block(beta_ar_hs, ar - t + 2, 1, t - 1, N_series), block(delta, 1, 1, t - 1, N_series), nu_trend[t]);
      } else {
        delta[t] = make_delta_t(alpha_ar, beta_ar_hs, block(delta, t - ar, 1, ar, N_series), nu_trend[t]);
      }
    }
  } else {
    row_vector[N_series] beta_ar_tmp = beta_ar_hs[1];
    delta[1] = make_delta_t_ar1(alpha_ar, beta_ar_tmp, delta_t0[1], nu_trend[1]); 
    for (t in 2:(N_periods-1)) delta[t] = make_delta_t_ar1(alpha_ar, beta_ar_tmp, delta[t-1], nu_trend[t]); 
  }


  // ----- SEASONALITY ------
  for (ss in 1:N_seasonality) {
    int periodicity = s[ss];
    
    // Apply shrinkage prior
    theta_season_hs[ss] = apply_hs_prior_v(theta_season[ss], tau0_theta_season, sigma_y, tau_theta_season, lambda_m_theta_season[ss], c_theta_season);
    
    tau_s[ss][1] = w_t[ss][1];
    for (t in 1:(periodicity-1)) {
      tau_s[ss][t] = w_t[ss][t];
    }
    for (t in periodicity:(N_periods-1)) {
      matrix[periodicity - 1, N_series] past_seasonality = block(tau_s[ss], t - periodicity + 1, 1, periodicity-1, N_series);
      
      for (d in 1:N_series) {
        tau_s[ss][t, d] = -sum(col(past_seasonality, d));
      }
      tau_s[ss][t] += w_t[ss][t];
    }
    if (ss == 1) tau = tau_s[1];
    else tau += tau_s[ss];
  }

    
  // ----- CYCLICALITY ------
  omega[1] = kappa[1];
  omega_star[1] = kappa_star[1]; 
  for (t in 2:(N_periods-1)) {
    omega[t] = (rho_cos_lambda .* omega[t - 1]) + (rho_sin_lambda .* omega_star[t-1]) + kappa[t];
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
  log_prices_hat[1] = starting_prices; 
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
  to_vector(delta_t0) ~ normal(0, inv_period_scale); 
  to_vector(alpha_ar) ~ normal(0, inv_period_scale); 
  hs_prior_lp(to_vector(beta_ar), tau_beta_ar, lambda_m_beta_ar, c_beta_ar, inv_period_scale, nu);
  to_vector(theta_ar) ~ cauchy(0, inv_period_scale); 
  L_omega_ar ~ lkj_corr_cholesky(1);

  // SEASONALITY
  for (ss in 1:N_seasonality) {
    hs_prior_lp(theta_season[ss], tau_theta_season, lambda_m_theta_season[ss], c_theta_season, inv_period_scale, nu);
  }
  
  // CYCLICALITY
  lambda ~ uniform(0, pi());
  rho ~ uniform(0, 1);
  hs_prior_lp(theta_cycle, tau_theta_cycle, lambda_m_theta_cycle, c_theta_cycle, inv_period_scale, nu);

  // REGRESSION
  hs_prior_lp(to_vector(beta_xi), tau_beta_xi, lambda_m_beta_xi, c_beta_xi, inv_period_scale, nu);

  // INNOVATIONS
  omega_garch ~ cauchy(0, inv_period_scale);
  hs_prior_lp(to_vector(beta_p), tau_beta_p, lambda_m_beta_p, c_beta_p, inv_period_scale, nu);
  hs_prior_lp(to_vector(beta_q), tau_beta_q, lambda_m_beta_q, c_beta_q, inv_period_scale, nu);
  L_omega_garch ~ lkj_corr_cholesky(1);

  // ----- TIME SERIES ------
  to_vector(starting_prices) ~ uniform(min_price, max_price); 
  nu_trend ~ multi_normal_cholesky(zero_vector, L_Omega_ar);
  for (t in 1:(N_periods-1)) {
    for (ss in 1:N_seasonality) w_t[ss, t] ~ normal(zero_vector, theta_season_hs[ss]);
    kappa[t] ~ normal(zero_vector, theta_cycle_hs);
    kappa_star[t] ~ normal(zero_vector, theta_cycle_hs);
    epsilon[t] ~ multi_normal_cholesky(zero_vector, make_L(theta[t], L_omega_garch));
  }

  // ----- OBSERVATIONS ------
  sigma_y ~ cauchy(0, 0.01);
  price_error ~ normal(0, inv_weights * sigma_y);
}

generated quantities {
  matrix[periods_to_predict, N_series]             log_predicted_prices; 
  matrix[periods_to_predict, N_series]             delta_hat; // Trend at time t
  matrix[periods_to_predict, N_series]             tau_hat[N_seasonality]; // Seasonality at time t
  matrix[periods_to_predict, N_series]             tau_hat_all;
  matrix[periods_to_predict, N_series]             omega_hat; // Cyclicality at time t
  matrix[periods_to_predict, N_series]             omega_star_hat; // Anti-cyclicality at time t
  matrix[periods_to_predict, N_series]             theta_hat; // Conditional variance of innovations 
  matrix[periods_to_predict, N_series]             epsilon_hat; 
  matrix[periods_to_predict, N_series]             nu_ar_hat; 
  matrix[periods_to_predict, N_series]             kappa_hat;
  matrix[periods_to_predict, N_series]             kappa_star_hat; 
  matrix[periods_to_predict, N_series]             w_t_hat[N_seasonality];
  matrix[N_series, N_series]                       trend_corr = crossprod(L_omega_ar);
  matrix[N_series, N_series]                       innovation_corr = crossprod(L_omega_garch);
  
  {
    matrix[N_series, N_series] diag_cycle = diag_matrix(theta_cycle_hs);
    matrix[N_series, N_series] diag_season[N_seasonality]; 
    vector[N_series] zero_t = zero_vector'; 
    
    for (ss in 1:N_seasonality) diag_season[ss] = diag_matrix(theta_season_hs[ss]);
    
    for (t in 1:periods_to_predict) {
      nu_ar_hat[t] = multi_normal_cholesky_rng(zero_t, L_Omega_ar)';
      kappa_hat[t] = multi_normal_rng(zero_t, diag_cycle)';
      kappa_star_hat[t] = multi_normal_rng(zero_t, diag_cycle)';
      for (ss in 1:N_seasonality) w_t_hat[ss][t] = multi_normal_rng(zero_t, diag_season[ss])';
    }
  }

  
  // TREND
  if (ar > 1) {
    for (t in 1:periods_to_predict) {
      if (t == 1) {
        delta_hat[1] = make_delta_t(alpha_ar, beta_ar_hs, block(delta, N_periods - ar, 1, ar, N_series), nu_ar_hat[1]);
      } else if (t <= ar) {
        int periods_forward = t - 1;
        int periods_back = ar - periods_forward;
        int start_period = N_periods- 1 - periods_back; 
        delta_hat[t] = make_delta_t(alpha_ar, beta_ar_hs, append_row(block(delta, start_period, 1, periods_back, N_series), 
                                                                        block(delta_hat, 1, 1, periods_forward, N_series)), nu_ar_hat[t]);
      } else {
        delta_hat[t] = make_delta_t(alpha_ar, beta_ar_hs, block(delta_hat, t - ar, 1, ar, N_series), nu_ar_hat[t]);
      }
    }
  } else {
    row_vector[N_series] beta_ar_tmp = beta_ar_hs[1];
    delta_hat[1] = make_delta_t_ar1(alpha_ar, beta_ar_tmp, delta[N_periods-1], nu_ar_hat[1]);
    for (t in 2:periods_to_predict) delta_hat[t] = make_delta_t_ar1(alpha_ar, beta_ar_tmp, delta_hat[t-1], nu_ar_hat[t]);
  }

  
  // SEASONALITY
  for (ss in 1:N_seasonality) {
    int periodicity = s[ss];
    for (t in 1:(periods_to_predict)) {
      matrix[periodicity - 1, N_series] prior_tau;
      
      if (t == 1) {
        prior_tau = block(tau_s[ss], N_periods - periodicity + 1, 1, periodicity - 1, N_series);
      } else if (t < periodicity) {
        prior_tau = append_row(
          block(tau_hat[ss], 1, 1, t-1, N_series), 
          block(tau_s[ss], N_periods - 1 - (periodicity - 1 - (t-1)), 1, periodicity - 1 - (t-1), N_series)
        );
      } else {
        prior_tau = block(tau_hat[ss], t - periodicity + 1, 1, periodicity - 1, N_series); 
      }
      
      for (d in 1:N_series) {
        tau_hat[ss][t, d] = -sum(col(prior_tau, d));
      }
      tau_hat[ss][t] += w_t_hat[ss][t]; 
    }  
    if (ss == 1) tau_hat_all = tau_hat[ss];
    else tau_hat_all += tau_hat[ss];
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
      q_component = columns_dot_product(beta_q_hs, square(append_row(
        block(epsilon, start_period, 1, periods_back, N_series),
        block(epsilon_hat, 1, 1, periods_forward, N_series)
      ))); 
    } else {
      q_component = columns_dot_product(beta_q_hs, square(block(epsilon_hat, t - q, 1, q, N_series)));
    }
    
    theta_hat[t] = omega_garch + p_component + q_component;
    epsilon_hat[t] = multi_normal_cholesky_rng(zero_vector', make_L(theta_hat[t], L_omega_garch))';
  }
  
  {
    matrix[periods_to_predict, N_series] price_changes = delta_hat + tau_hat_all + omega_hat + epsilon_hat;
    
    log_predicted_prices[1] = log_prices_hat[N_periods] + price_changes[1];
    for (t in 2:periods_to_predict) {
      log_predicted_prices[t] = log_predicted_prices[t-1] + price_changes[t];
    }
  }

}

