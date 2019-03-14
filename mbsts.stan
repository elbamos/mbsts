functions {
  
  matrix make_L(row_vector theta, matrix Omega) {
    return diag_pre_multiply(sqrt(theta), Omega);
  }
  
  // Linear Trend 
  row_vector make_delta_t(row_vector alpha_trend, matrix beta_trend, matrix delta_past, row_vector nu) {
      return alpha_trend + columns_dot_product(beta_trend, delta_past) + nu;
  }
  
  // Enforce stationarity of lineartrend and GARCH
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

}

data { 
  int<lower=2> N; // Number of price points
  int<lower=2> N_series; // Number of price series
  int<lower=2> N_periods; // Number of periods
  int<lower=1> N_features; // Number of features in the regression
  
  // Parameters controlling the model 
  int<lower=2> periods_to_predict;
  int<lower=1> ar; // AR period for the trend
  int<lower=1> p; // GARCH
  int<lower=1> q; // GARCH
  int<lower=1> N_seasonality;
  int<lower=1> s[N_seasonality]; // seasonality 
  real<lower=1> period_scale; 
  real<lower=3> cyclicality_prior; // Prior estimate of the number of periods in the business cycle 
  int<lower=0> corr_prior; 
  
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
  real                                min_beta_ar = ar == 1 ? 0 : -1;
  real                                lambda_mean = 2 / cyclicality_prior; 
  real                                lambda_a = -lambda_mean * 2 / (lambda_mean - 1); 
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
  
  // TREND delta_t
  matrix[1, N_series]                                 delta_t0; // Trend at time 0
  row_vector[N_series]                                alpha_ar; // long-term trend
  // Note that beta_ar is converted to beta_ar_c to enforce stationarity
  matrix<lower=min_beta_ar,upper=1>[ar, N_series]     beta_ar; // Learning rate of trend
  row_vector[N_series]                                nu_trend[N_periods-1]; // Random changes in trend
  row_vector<lower=0>[N_series]                       theta_ar; // Variance in changes in trend
  cholesky_factor_corr[N_series]                      L_omega_ar; // Correlations among trend changes
  
  // SEASONALITY
  matrix[N_periods-1+max_s, N_series]                 w_t[N_seasonality]; // Random variation in seasonality
  vector<lower=0>[N_series]                           theta_season[N_seasonality]; // Variance in seasonality

  // CYCLICALITY
  row_vector<lower=0, upper=pi()>[N_series]           lambda; // Frequency
  row_vector<lower=0, upper=1>[N_series]              rho; // Damping factor
  vector<lower=0>[N_series]                           theta_cycle; // Variance in cyclicality
  row_vector[N_series]                                kappa[N_periods - 1];  // Random changes in cyclicality
  row_vector[N_series]                                kappa_star[N_periods - 1]; // Random changes in counter-cycle
  
  // REGRESSION
  matrix[N_features, N_series]                        beta_xi; // Coefficients of the regression parameters
  
  // INNOVATIONS
  matrix[N_periods-1, N_series]                       epsilon; // Innovations
  row_vector<lower=0>[N_series]                       omega_garch; // Baseline volatility of innovations
  // Note that beta_p and q are converted to beta_p_c and q_c to enforce stationarity
  matrix<lower=0,upper=1>[p, N_series]                beta_p; // Univariate GARCH coefficients on prior volatility
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
  matrix[N_periods, N_series]                         xi = x * beta_xi; // Predictors
  vector[N]                                           log_y_hat; 
  matrix[N_series, N_series]                          L_Omega_ar = make_L(theta_ar, L_omega_ar);
  row_vector[N_series] rho_cos_lambda = rho .* cos(lambda); 
  row_vector[N_series] rho_sin_lambda = rho .* sin(lambda); 
  // Constrain to stationarity
  matrix[ar, N_series]                                beta_ar_c = ar == 1 ? beta_ar : constrain_stationary(beta_ar);
  matrix[p, N_series]                                 beta_p_c = p == 1 ? beta_p : constrain_stationary(beta_p);

  // TREND
  delta[1] = make_delta_t(alpha_ar, block(beta_ar, ar, 1, 1, N_series), delta_t0, nu_trend[1]);
  for (t in 2:(N_periods-1)) {
    if (t <= ar) {
      delta[t] = make_delta_t(alpha_ar, 
                              constrain_stationary(block(beta_ar, ar - t + 2, 1, t - 1, N_series)), 
                              block(delta, 1, 1, t - 1, N_series), nu_trend[t]);
    } else {
      delta[t] = make_delta_t(alpha_ar, beta_ar_c, block(delta, t - ar, 1, ar, N_series), nu_trend[t]);
    }
  }



  // ----- SEASONALITY ------
  for (ss in 1:N_seasonality) {
    int periodicity = s[ss] - 1;
    matrix[N_periods - 1 + periodicity, N_series]  tau_s_temp; 
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
                                          constrain_stationary(block(beta_p, p - t + 2, 1, t - 1, N_series)),
                                          block(theta, 1, 1, t - 1, N_series)
                                          );
      } else {
        p_component = columns_dot_product(beta_p_c, block(theta, t - p, 1, p, N_series));
      }
      
      if (t <= q) {
        q_component = columns_dot_product(block(beta_q, q - t + 2, 1, t - 1, N_series), block(epsilon_squared, 1, 1, t - 1, N_series));
      } else {
        q_component = columns_dot_product(beta_q, block(epsilon_squared, t - q, 1, q, N_series));
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
  to_vector(alpha_ar) ~ normal(0, inv_period_scale); 
  to_vector(delta_t0) ~ normal(alpha_ar, inv_period_scale); 
  // Jones (1984) Prior on the partial autocorrelations
  // Sets a uniform prior on partial autocorrelations 
  for (ss in 1:N_series) {
    .5 + (col(beta_ar, ss) / 2) ~ beta(beta_ar_alpha, beta_ar_beta); 
  }
  to_vector(beta_ar) ~ cauchy(0, 0.3); 
  to_vector(theta_ar) ~ cauchy(0, inv_period_scale); 
  L_omega_ar ~ lkj_corr_cholesky(corr_prior);

  // SEASONALITY
  for (ss in 1:N_seasonality) {
    theta_season[ss] ~ cauchy(0, inv_period_scale); 
    for (t in 1:max_s) w_t[ss, t] ~ normal(zero_vector, theta_season[ss]);
  }
  
  // CYCLICALITY
  (lambda / pi()) ~ beta(lambda_a, 2); 
  rho ~ normal(0, 1);
  theta_cycle ~ cauchy(0, inv_period_scale);

  // REGRESSION
  to_vector(beta_xi) ~ cauchy(0, inv_period_scale); 

  // INNOVATIONS
  omega_garch ~ normal(0, inv_period_scale);
  to_vector(beta_p) ~ cauchy(0, .3);
  to_vector(beta_q) ~ cauchy(0, .3); 
  L_omega_garch ~ lkj_corr_cholesky(corr_prior);

  // ----- TIME SERIES ------
  // Time series
  to_vector(starting_prices) ~ uniform(min_price, max_price); 
  nu_trend   ~ multi_normal_cholesky(zero_vector, L_Omega_ar);
  for (t in 1:(N_periods-1)) {
    for (ss in 1:N_seasonality) w_t[ss][t] ~ normal(zero_vector, theta_season[ss]);
    epsilon[t]    ~ multi_normal_cholesky(zero_vector, make_L(theta[t], L_omega_garch));
    kappa[t]      ~ normal(zero_vector, theta_cycle);
    kappa_star[t] ~ normal(zero_vector, theta_cycle);
  }
  for (t in N_periods:(N_periods + max_s - 1)) {
    for (ss in 1:N_seasonality) w_t[ss][t] ~ normal(zero_vector, theta_season[ss]);
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
    kappa_hat[t] = multi_normal_rng(zero_vector_r, diag_matrix(theta_cycle))';
    kappa_star_hat[t] = multi_normal_rng(zero_vector_r, diag_matrix(theta_cycle))';
    for (ss in 1:N_seasonality) w_t_hat[ss][t] = multi_normal_rng(zero_vector_r, diag_matrix(theta_season[ss]))';
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
      q_component = columns_dot_product(beta_q, square(block(epsilon_temp, t, 1, q, N_series)));
      
      theta_temp[t + p] = omega_garch + p_component + q_component;
      epsilon_temp[t + q] = multi_normal_cholesky_rng(zero_vector_r, make_L(theta_temp[t + p], L_omega_garch))';
    }
    
    theta_hat = block(theta_temp, p + 1, 1, periods_to_predict, N_series);
    epsilon_hat = block(epsilon_temp, q + 1, 1, periods_to_predict, N_series); 
  }
  
  log_prices_hat[1] = log_prices[N_periods] + delta_hat[1] + tau_hat_all[1] + omega_hat[1] + epsilon_hat[1];
  for (t in 2:periods_to_predict) {
    log_prices_hat[t] = log_prices_hat[t-1] + delta_hat[t] + tau_hat_all[t] + omega_hat[t] + epsilon_hat[t];
  }
}

