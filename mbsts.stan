functions {
  
  matrix make_L(row_vector theta, matrix Omega) {
    return diag_pre_multiply(sqrt(theta), Omega);
  }
  
  // Linear Trend 
  row_vector make_delta_t(row_vector alpha_trend, matrix beta_trend, matrix delta_past, row_vector nu) {
      return alpha_trend + columns_dot_product(beta_trend, delta_past) + nu;
  }
  
  // ---- Stationarity ----
  row_vector pacf_to_acf(vector x) {
    int n = num_elements(x);
    matrix[n, n] y = diag_matrix(x);
    row_vector[n] out; 

    for (k in 2:n) {
      for (i in 1:(k - 1)) {
        y[k, i] = y[k - 1, i] - x[k] * y[k - 1, k - i];
      }
    }
    for (i in 1:n) {
      out[i] = y[n, n - i + 1];
    }
    return out;
  }
  
  matrix constrain_stationary(matrix x) {
    matrix[cols(x), rows(x)]  out;
    for (n in 1:cols(x)) {
      out[n] = pacf_to_acf(col(x, n)); 
    }
    return out';
  }
  
    // Jones (1984) prior on ar coefficients
  // Sets a uniform prior on partial autocorrelations
  real jonesprior_lpdf(matrix pacfs, vector beta_ar_alpha, vector beta_ar_beta) {
    real tag = 0;
    if (rows(pacfs) == 1) return 0; 
    else {
      matrix[rows(pacfs), cols(pacfs)] trans = (pacfs / 2.0) + 0.5; 
      for (i in 1:cols(pacfs)) {
        tag += beta_lpdf(col(trans, i) | beta_ar_alpha, beta_ar_beta);
      }
      // This isn't strictly necessary since its a constant...
      tag += log(0.5) * num_elements(pacfs); 
      return tag; 
    }
  }
  
  // ----- Sparsity -----
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
  real<lower=3> cyclicality_prior; // Prior estimate of the number of periods in the business cycle 
  int<lower=0> corr_prior; 
  
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
  vector[N_series]                    zero_vector_r = zero_vector';
  vector<lower=0>[N]                  inv_weights;
  real<lower=0>                       inv_period_scale = 1.0 / period_scale; 
  // ---- Sparsity ----
  real                                tau0_vector = get_tau0_start(m0, N_series, N_periods);
  real                                tau0_ar = get_tau0_start(m0, N_series * ar, N_periods);
  real                                tau0_beta_p = get_tau0_start(m0, N_series * p, N_periods);
  real                                tau0_beta_q = get_tau0_start(m0, N_series * q, N_periods);
  real                                tau0_xi = get_tau0_start(m0, N_series * N_features, N_periods);
  real                                tau0_theta_cycle = get_tau0_start(m0, N_series, N_periods);
  real                                tau0_theta_season = get_tau0_start(m0, N_series * N_seasonality, N_periods);
  // ----- Helpers -----
  real                                min_beta_ar = ar == 1 ? 0 : -1;
  real                                lambda_mean = 2 / cyclicality_prior; 
  real                                lambda_a = -lambda_mean * 2 / (lambda_mean - 1); 
  //real                                cyc_transform = log(1.0/pi()) * N_series; 
  int                                 max_s = max(s) - 1;
  // Priors for beta_ar partial autocorrelations
  vector<lower=0>[ar]                 beta_ar_alpha;
  vector<lower=0>[ar]                 beta_ar_beta;
  for (a in 1:ar) {
    beta_ar_alpha[ar - a + 1] = floor((a + 1.0)/2.0); 
    beta_ar_beta[ar - a + 1] = floor(a / 2.0) + 1; 
  }
  
  for (n in 1:N) {
    log_y[n] = log1p(y[n]);
    inv_weights[n] = 1.0 / weight[n];
  }
}


parameters {
  real<lower=0>                                       sigma_y; // observation variance
  
  // ---- TREND delta_t ----
  matrix[1, N_series]                                 delta_t0; // Trend at time 0
  row_vector[N_series]                                alpha_ar; // long-term trend
  // Hierarchical shrinkage prior on beta_trend 
  vector<lower=0>[ar * N_series]                      lambda_m_beta_ar; 
  real<lower=0>                                       c_beta_ar;
  real<lower=0>                                       tau_beta_ar;
  // Note that beta_ar is converted to beta_ar_c to enforce stationarity
  matrix<lower=min_beta_ar,upper=1>[ar, N_series]     beta_ar; // Learning rate of trend
  row_vector[N_series]                                nu_trend[N_periods-1]; // Random changes in trend
  row_vector<lower=0>[N_series]                       theta_ar; // Variance in changes in trend
  cholesky_factor_corr[N_series]                      L_omega_ar; // Correlations among trend changes
  
  // ---- SEASONALITY ----
  vector<lower=0>[N_series]                           lambda_m_theta_season[N_seasonality]; 
  real<lower=0>                                       c_theta_season;
  real<lower=0>                                       tau_theta_season;
  matrix[N_periods-1+max_s, N_series]                 w_t[N_seasonality]; // Random variation in seasonality
  vector<lower=0>[N_series]                           theta_season[N_seasonality]; // Variance in seasonality

  // ---- CYCLICALITY ----
  row_vector<lower=0, upper=pi()>[N_series]           lambda; // Frequency
  row_vector<lower=0, upper=1>[N_series]              rho; // Damping factor
  // Hierarchical shrinkage prior on beta_trend 
  vector<lower=0>[N_series]                           lambda_m_theta_cycle; 
  real<lower=0>                                       c_theta_cycle;
  real<lower=0>                                       tau_theta_cycle;
  vector<lower=0>[N_series]                           theta_cycle; // Variance in cyclicality
  row_vector[N_series]                                kappa[N_periods - 1];  // Random changes in cyclicality
  row_vector[N_series]                                kappa_star[N_periods - 1]; // Random changes in counter-cycle
  
  // ---- REGRESSION ----
  vector<lower=0>[N_features * N_series]              lambda_m_beta_xi; 
  real<lower=0>                                       c_beta_xi;
  real<lower=0>                                       tau_beta_xi;
  matrix[N_features, N_series]                        beta_xi; // Coefficients of the regression parameters
  
  // ---- INNOVATIONS ----
  matrix[N_periods-1, N_series]                       epsilon; // Innovations
  row_vector<lower=0>[N_series]                       omega_garch; // Baseline volatility of innovations
    // Hierarchical shrinkage prior on beta_p
  vector<lower=0>[p * N_series]                       lambda_m_beta_p; 
  real<lower=0>                                       c_beta_p;
  real<lower=0>                                       tau_beta_p;
  matrix<lower=0,upper=1>[p, N_series]                beta_p; // Univariate GARCH coefficients on prior volatility
  // Hierarchical shrinkage prior on beta_q
  vector<lower=0>[q * N_series]                       lambda_m_beta_q; 
  real<lower=0>                                       c_beta_q;
  real<lower=0>                                       tau_beta_q;
  matrix<lower=0,upper=1>[q, N_series]                beta_q; // Univariate GARCH coefficients on prior innovations
  cholesky_factor_corr[N_series]                      L_omega_garch; // Constant correlations among innovations 
  
  row_vector<lower=min_price,upper=max_price>[N_series] starting_prices;
}

transformed parameters {
  matrix[N_periods, N_series]                         log_prices; // Observable prices
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
  matrix[p, N_series] beta_p_hs = apply_hs_prior_m(beta_p, tau0_beta_p, sigma_y, tau_beta_p, lambda_m_beta_p, c_beta_p); 
  matrix[q, N_series] beta_q_hs = apply_hs_prior_m(beta_q, tau0_beta_q, sigma_y, tau_beta_q, lambda_m_beta_q, c_beta_q); 
  matrix[N_features, N_series] beta_xi_hs = apply_hs_prior_m(beta_xi, tau0_xi, sigma_y, tau_beta_xi, lambda_m_beta_xi, c_beta_xi); 
  vector[N_series] theta_cycle_hs = apply_hs_prior_v(theta_cycle, tau0_theta_cycle, sigma_y, tau_theta_cycle, lambda_m_theta_cycle, c_theta_cycle);
  vector[N_series] theta_season_hs[N_seasonality];

  // Retain calculation of xi
  matrix[N_periods, N_series]                         xi = x * beta_xi_hs; // Predictors

  // Constrain to stationarity
  matrix[ar, N_series]     beta_ar_c = ar == 1 ? beta_ar_hs : constrain_stationary(beta_ar_hs);
  matrix[p, N_series]      beta_p_c = p == 1 ? beta_p_hs : constrain_stationary(beta_p_hs);

  // TREND
  delta[1] = make_delta_t(alpha_ar, block(beta_ar_c, ar, 1, 1, N_series), delta_t0, nu_trend[1]);
  for (t in 2:(N_periods-1)) {
    if (t <= ar) {
      delta[t] = make_delta_t(alpha_ar, 
                              constrain_stationary(block(beta_ar_c, ar - t + 2, 1, t - 1, N_series)), 
                              block(delta, 1, 1, t - 1, N_series), nu_trend[t]);
    } else {
      delta[t] = make_delta_t(alpha_ar, beta_ar_c, block(delta, t - ar, 1, ar, N_series), nu_trend[t]);
    }
  }

  // ----- SEASONALITY ------
  for (ss in 1:N_seasonality) {
    int periodicity = s[ss] - 1;
    matrix[N_periods - 1 + periodicity, N_series]  tau_s_temp; 
    
    // Apply shrinkage prior
    theta_season_hs[ss] = apply_hs_prior_v(theta_season[ss], tau0_theta_season, sigma_y, tau_theta_season, lambda_m_theta_season[ss], c_theta_season);
    
    tau_s_temp[1] = w_t[ss][1];
    
    for (t in 2:(N_periods -1 + periodicity)) {
      for (d in 1:N_series) tau_s_temp[t, d] = -sum(sub_col(tau_s_temp, max(1, t - periodicity), d, min(periodicity, t-1)));
      tau_s_temp[t] += w_t[ss][t];
    }
    
    tau_s[ss] = block(tau_s_temp, periodicity + 1, 1, N_periods - 1, N_series); 
    if (ss == 1) tau = tau_s[ss];
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
        p_component = columns_dot_product(
                                          constrain_stationary(block(beta_p_hs, p - t + 2, 1, t - 1, N_series)),
                                          block(theta, 1, 1, t - 1, N_series)
                                          );
      } else {
        p_component = columns_dot_product(beta_p_c, block(theta, t - p, 1, p, N_series));
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
  log_prices[1] = starting_prices; 
  for (t in 2:N_periods) {
    log_prices[t] = log_prices[t-1] + delta[t-1] + tau[t-1] + omega[t-1] + xi[t-1] + epsilon[t-1];
  }
  
  for (n in 1:N) {
    log_y_hat[n] = log_prices[period[n], series[n]];
  }
}


model {
  vector[N] price_error = log_y - log_y_hat;
  
  // ----- PRIORS ------
  // TREND 
  to_vector(alpha_ar) ~ student_t(1, 0, inv_period_scale); 
  to_vector(delta_t0) ~ normal(alpha_ar, inv_period_scale); 
  hs_prior_lp(to_vector(beta_ar), tau_beta_ar, lambda_m_beta_ar, c_beta_ar, 1, nu);
  beta_ar ~ jonesprior(beta_ar_alpha, beta_ar_beta); 
  L_omega_ar ~ lkj_corr_cholesky(corr_prior);

  // SEASONALITY
  for (ss in 1:N_seasonality) {
    hs_prior_lp(theta_season[ss], tau_theta_season, lambda_m_theta_season[ss], c_theta_season, inv_period_scale, nu);
    for (t in 1:max_s) w_t[ss, t] ~ normal(zero_vector, theta_season[ss]);
  }
  
  // CYCLICALITY
  (lambda / pi()) ~ beta(lambda_a, 2); 
  rho ~ uniform(0, 1);
  hs_prior_lp(theta_cycle, tau_theta_cycle, lambda_m_theta_cycle, c_theta_cycle, inv_period_scale, nu);
  // The log abs det of the transform above isn't necessary as its a constant
  // target += cyc_transform; 

  // REGRESSION
  hs_prior_lp(to_vector(beta_xi), tau_beta_xi, lambda_m_beta_xi, c_beta_xi, inv_period_scale, nu);

  // INNOVATIONS
  omega_garch ~ student_t(1, 0, inv_period_scale);
  hs_prior_lp(to_vector(beta_p), tau_beta_p, lambda_m_beta_p, c_beta_p, 1, nu);
  hs_prior_lp(to_vector(beta_q), tau_beta_q, lambda_m_beta_q, c_beta_q, 1, nu);
  L_omega_garch ~ lkj_corr_cholesky(corr_prior);

  // ----- TIME SERIES ------
  to_vector(starting_prices) ~ uniform(min_price, max_price); 
  nu_trend   ~ multi_normal_cholesky(zero_vector, L_Omega_ar);
  for (t in 1:(N_periods-1)) {
    for (ss in 1:N_seasonality) w_t[ss][t] ~ normal(zero_vector, theta_season_hs[ss]);
    epsilon[t]    ~ multi_normal_cholesky(zero_vector, make_L(theta[t], L_omega_garch));
    kappa[t]      ~ normal(zero_vector, theta_cycle_hs);
    kappa_star[t] ~ normal(zero_vector, theta_cycle_hs);
  }
  for (t in N_periods:(N_periods + max_s - 1)) {
    for (ss in 1:N_seasonality) w_t[ss][t] ~ normal(zero_vector, theta_season_hs[ss]);
  } 

  // ----- OBSERVATIONS ------
  sigma_y ~ cauchy(0, 0.01);
  price_error ~ normal(0, inv_weights * sigma_y);
}

generated quantities {
  matrix[periods_to_predict, N_series]             log_prices_hat; 
  matrix[periods_to_predict, N_series]             delta_hat; // Expected trend at time t
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
  
  for (t in 1:periods_to_predict) {
    nu_ar_hat[t] = multi_normal_cholesky_rng(zero_vector_r, L_Omega_ar)';
    kappa_hat[t] = multi_normal_rng(zero_vector_r, diag_matrix(theta_cycle_hs))';
    kappa_star_hat[t] = multi_normal_rng(zero_vector_r, diag_matrix(theta_cycle_hs))';
    for (ss in 1:N_seasonality) w_t_hat[ss][t] = multi_normal_rng(zero_vector_r, diag_matrix(theta_season_hs[ss]))';
  }

  
  // TREND
  {
    matrix[ar + periods_to_predict, N_series] delta_temp = append_row(
      block(delta, N_periods - ar, 1, ar, N_series), 
      rep_matrix(0, periods_to_predict, N_series)
    ); 
    
    for (t in 1:periods_to_predict) delta_temp[ar + t] = make_delta_t(alpha_ar, beta_ar_c, 
                                                                      block(delta_temp, t, 1, ar, N_series), 
                                                                      nu_ar_hat[1]);
                                                
    delta_hat = block(delta_temp, ar + 1, 1, periods_to_predict, N_series); 
  } 
  
  // SEASONALITY
  for (ss in 1:N_seasonality) {
    int periodicity = s[ss] - 1;
    matrix[periodicity + periods_to_predict, N_series] tau_temp = append_row(
      block(tau_s[ss], N_periods - periodicity, 1, periodicity, N_series), 
      rep_matrix(0, periods_to_predict, N_series)
    ); 
    
    for (t in 1:(periods_to_predict)) {
      for (d in 1:N_series) tau_temp[periodicity + t, d] = -sum(sub_col(tau_temp, t, d, periodicity));
      tau_temp[periodicity + t] += w_t_hat[ss][t]; 
    }  
    if (ss == 1) tau_hat_all = block(tau_temp, periodicity + 1, 1, periods_to_predict, N_series);
    else tau_hat_all += block(tau_temp, periodicity + 1, 1, periods_to_predict, N_series);
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
  {
    matrix[p + periods_to_predict, N_series] theta_temp = append_row(
      block(theta, N_periods - p, 1, p, N_series), 
      rep_matrix(0, periods_to_predict, N_series)
    );
    matrix[q + periods_to_predict, N_series] epsilon_temp = append_row(
      block(epsilon, N_periods - q, 1, q, N_series), 
      rep_matrix(0, periods_to_predict, N_series)
    ); 
    
    for (t in 1:periods_to_predict) {
      row_vector[N_series]  p_component; 
      row_vector[N_series]  q_component;      
      
      p_component = columns_dot_product(beta_p_c, block(theta_temp, t, 1, p, N_series));
      q_component = columns_dot_product(beta_q_hs, square(block(epsilon_temp, t, 1, q, N_series)));

      theta_temp[t + p] = omega_garch + p_component + q_component;
      epsilon_temp[t + q] = multi_normal_cholesky_rng(zero_vector_r, make_L(theta_temp[t + q], L_omega_garch))';
    }
    
    theta_hat = block(theta_temp, p + 1, 1, periods_to_predict, N_series);
    epsilon_hat = block(epsilon_temp, q + 1, 1, periods_to_predict, N_series); 
  }
  
  {
    matrix[periods_to_predict, N_series] price_changes = delta_hat + tau_hat_all + omega_hat + epsilon_hat;
    log_prices_hat[1] = log_prices[N_periods] + price_changes[1];
    for (t in 2:periods_to_predict) {
      log_prices_hat[t] = log_prices_hat[t-1] + price_changes[t];
    }
  }

}

