/*
Potential issues:
---- Hierarchical shrinkage ----
- How do we choose the slab_scale? currently using 1
- Which volatility to use in calculating tau0? currently using sigma_price
- For generating the parameters, the original code used a constant sigma to draw the params, then multiplied them by the shrinkage prior; this code samples against the shrinkage prior directly. The former may have better geometry

---- Cyclicality ---- 
I'm not 100 % confident that I'm correctly interpreting the formula

---- Other ---- 
- Do we want shrinkage priors on the trend, seasonality, or price shocks themselves?
- Prior on seasonality?
- Prior on lambda for cyclicality? 
- Prior on the damping factor for cyclicality?
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
  void apply_hs_prior_lp(vector param, real target_sparsity, int N, real sigma, real slab_scale, real tau, vector lambda_m, real c2_tilde, real nu) {
    int M = num_elements(param);
    real m0 =  M * target_sparsity;
    real tau0 = (m0 / (M - m0)) * (sigma / sqrt(1.0 * N));
    vector[M] lambda_m_tilde;
    real c2; 
    
    tau ~ cauchy(0, 1);
    c2_tilde ~ inv_gamma(nu/2, nu / 2);
    lambda_m ~ cauchy(0, 1);
    c2 = c2_tilde * slab_scale;
    lambda_m_tilde = sqrt(c2 * square(lambda_m) ./ (c2 + square(tau * tau0) * square(lambda_m)));
    
    param ~ normal(0, tau * tau0 * lambda_m_tilde);
  }

}

data { 
  int<lower=2> N; // Number of price points
  int<lower=2> D; // Number of price series
  int<lower=2> P; // Number of periods
  int<lower=1> F; // Number of features in the regression
  
  // Parameters controlling the time series 
  int<lower=2> periods_to_predict;
  int<lower=1> ar; // AR period for the trend
  int<lower=1> p; // GARCH
  int<lower=1> q; // GARCH
  int<lower=1> s[D]; // seasonality periods
  
  // Parameeters controlling sparse feature selection
  real<lower=0,upper=1>              m0; // Sparsity target; proportion of features we expect to be relevant
  int<lower=1>                       nu; // nu parameters for horseshoe prior
  
  // Data 
  vector<lower=0>[N]                  y;
  int<lower=1,upper=P>                period[N];
  int<lower=1,upper=D>                series[N];
  vector<lower=0>[N]                  quantity;
  matrix[P, F]                        x; // Regression predictors
  
  matrix[periods_to_predict, F]       x_predictive;
}

transformed data {
  vector<lower=0>[N]                  log_y;
  real<lower=0>                       min_price =  log1p(min(y));
  real<lower=0>                       max_price = log1p(max(y));
  row_vector[D]                       zero_vector = rep_row_vector(0, D);
  //vector[D]                           counts = rep_row_vector(0, D); 

  for (n in 1:N) {
    log_y[n] = log1p(y[n]);
    //counts[series[n]] += 1; // We keep track of counts to correctly calculate tau_0
    // for the hierarchical shrinkage prior
  }
}


parameters {
  real<lower=0>                                sigma_y; // observation variance
  
  // TREND delta_t
  matrix[1, D]                                 delta_t0; // Trend at time 0
  row_vector[D]                                alpha_trend; // long-term trend
  // Hierarchical shrinkage prior on beta_trend
  vector<lower=0>[ar * D]                      lambda_m_beta_trend; 
  real<lower=0>                                c_beta_trend;
  real<lower=0>                                tau_beta_trend;
  matrix<lower=0,upper=1>[ar, D]               beta_trend; // Learning rate of trend
  row_vector[D]                                nu_trend[P-1]; // Random changes in trend
  row_vector<lower=0>[D]                       theta_trend; // Variance in changes in trend
  cholesky_factor_corr[D]                      L_omega_trend; // Correlations among trend changes
  
  // SEASONALITY
  row_vector[D]                                w_t[P-1]; // Random variation in seasonality
  vector<lower=0>[D]                           theta_season; // Variance in seasonality

  // CYCLICALITY
  row_vector<lower=0, upper=pi()>[D]           lambda; // Frequency
  row_vector<lower=0, upper=1>[D]              rho; // Damping factor
  vector<lower=0>[D]                           theta_cycle; // Variance in cyclicality
  matrix[P - 1, D]                             kappa;  // Random changes in cyclicality
  matrix[P - 1, D]                             kappa_star; // Random changes in counter-cyclicality
  
  // REGRESSION
  vector<lower=0>[F * D]                       lambda_m_beta_xi; 
  real<lower=0>                                c_beta_xi;
  real<lower=0>                                tau_beta_xi;
  matrix[F, D]                                 beta_xi; // Coefficients of the regression parameters
  
  // INNOVATIONS
  matrix[P-1, D]                               epsilon; // Innovations
  row_vector[D]                                omega_garch; // Baseline volatility of innovations
    // Hierarchical shrinkage prior on beta_p
  vector<lower=0>[p * D]                       lambda_m_beta_p; 
  real<lower=0>                                c_beta_p;
  real<lower=0>                                tau_beta_p;
  matrix[p, D]                                 beta_p; // Univariate GARCH coefficients on prior volatility
  // Hierarchical shrinkage prior on beta_q
  vector<lower=0>[q * D]                       lambda_m_beta_q; 
  real<lower=0>                                c_beta_q;
  real<lower=0>                                tau_beta_q;
  matrix[q, D]                                 beta_q; // Univariate GARCH coefficients on prior innovations
  cholesky_factor_corr[D]                      L_omega_garch; // Constant correlations among innovations 
  
  row_vector<lower=0>[D]                       starting_prices;
}

transformed parameters {
  matrix[P, D]                              log_prices_hat; // Observable prices
  matrix[P-1, D]                            delta; // Trend at time t
  matrix[P-1, D]                            tau; // Seasonality at time t
  matrix[P-1, D]                            omega; // Cyclicality at time t
  matrix[P-1, D]                            omega_star; // Anti-cyclicality at time t
  matrix[P-1, D]                            theta; // Conditional variance of innovations 
  vector[N]                                 log_y_hat; 
  matrix[D, D]                              L_Omega_trend = make_L(theta_trend, L_omega_trend);
  
  // TREND
  delta[1] = make_delta_t(alpha_trend, block(beta_trend, ar, 1, 1, D), delta_t0, nu_trend[1]);
  for (t in 2:(P-1)) {
    if (t <= ar) {
      delta[t] = make_delta_t(alpha_trend, block(beta_trend, ar - t + 2, 1, t - 1, D), block(delta, 1, 1, t - 1, D), nu_trend[t]);
    } else {
      delta[t] = make_delta_t(alpha_trend, beta_trend, block(delta, t - ar, 1, ar, D), nu_trend[t]);
    }
  }
  
  // SEASONALITY
  tau[1] = -w_t[1];
  for (t in 2:(P-1)) {
    for (d in 1:D) {
      tau[t, d] = -sum(sub_col(tau, max(1, t - 1 - s[d]), d, min(s[d], t - 1)));
    }
    tau[t] += w_t[t];
  }
  
  // Cyclicality
  omega[1] = kappa[1];
  omega_star[1] = kappa_star[1]; 
  {
    row_vector[D] rho_cos_lambda = rho .* cos(lambda); 
    row_vector[D] rho_sin_lambda = rho .* sin(lambda); 
    for (t in 2:(P-1)) {
      omega[t] = (rho_cos_lambda .* omega[t - 1]) + (rho_sin_lambda .* omega_star[t-1]) + kappa[t];
      # TODO: Confirm that the negative only applies to the first factor not both
      omega_star[t] = - (rho_sin_lambda .* omega[t - 1]) + (rho_cos_lambda .* omega_star[t-1]) + kappa_star[t];
    }
  }

  
  // Univariate GARCH
  theta[1] = omega_garch; 
  {
    matrix[P-1, D] epsilon_squared = square(epsilon);
    
    for (t in 2:(P-1)) {
      row_vector[D]  p_component; 
      row_vector[D]  q_component; 
      
      if (t <= p) {
        p_component = columns_dot_product(block(beta_p, p - t + 2, 1, t - 1, D), block(theta, 1, 1, t - 1, D));
      } else {
        p_component = columns_dot_product(beta_p, block(theta, t - p, 1, p, D));
      }
      
      if (t <= q) {
        q_component = columns_dot_product(block(beta_q, q - t + 2, 1, t - 1, D), block(epsilon_squared, 1, 1, t - 1, D));
      } else {
        q_component = columns_dot_product(beta_q, block(epsilon_squared, t - q, 1, q, D));
      }
      
      theta[t] = omega_garch + p_component + q_component;
    }
  }
 
  
  
  {
    matrix[P, D] xi = beta_xi * x;
    
    log_prices_hat[1] = starting_prices + xi[1]; 
    for (t in 2:(P-1)) {
      log_prices_hat[t] = log_prices_hat[t-1] + delta[t] + tau[t] + omega[t] + xi[t] + epsilon[t];
    }
  }
  
  log_y_hat = to_vector(log_prices_hat[period, series]);
}


model {
  vector[N] price_error = log_y - log_y_hat;
  
  // ----- PRIORS ------
  // TREND 
  to_vector(delta_t0) ~ normal(0, 1); 
  to_vector(alpha_trend) ~ normal(0, 1); 
  apply_hs_prior_lp(to_vector(beta_trend), m0, N, sigma_y, 1, tau_beta_trend, lambda_m_beta_trend, c_beta_trend, nu);
  to_vector(theta_trend) ~ cauchy(0, 1); 
  L_omega_trend ~ lkj_corr_cholesky(1);
  
  // SEASONALITY
  theta_season ~ cauchy(0, 1); 

  // CYCLICALITY
  lambda ~ uniform(0, pi());
  rho ~ uniform(0, 1);
  theta_cycle ~ cauchy(0, 1);
  
  // REGRESSION
  // TODO: Is sigma_y the correct volatility to use here?
  // TODO: How do we determine the slab scale? 
  apply_hs_prior_lp(to_vector(beta_xi), m0, N, sigma_y, 1, tau_beta_xi, lambda_m_beta_xi, c_beta_xi, nu);

  // INNOVATIONS
  omega_garch ~ cauchy(0, 1);
  apply_hs_prior_lp(to_vector(beta_p), m0, N, sigma_y, 1, tau_beta_p, lambda_m_beta_p, c_beta_p, nu);
  apply_hs_prior_lp(to_vector(beta_q), m0, N, sigma_y, 1, tau_beta_q, lambda_m_beta_q, c_beta_q, nu);
  L_omega_garch ~ lkj_corr_cholesky(1);

  // Time series
  to_vector(starting_prices) ~ gamma(2, 0); 
  nu_trend ~ multi_normal_cholesky(zero_vector, L_Omega_trend);
  for (t in 1:(P-1)) {
    w_t[t] ~ normal(zero_vector, theta_season);
    kappa[t] ~ normal(zero_vector, theta_cycle);
    kappa_star[t] ~ normal(zero_vector, theta_cycle);
    epsilon[t] ~ multi_normal_cholesky(zero_vector, make_L(theta[t], L_omega_garch));
  }

  // Observations
  sigma_y ~ cauchy(0, 0.01);
  price_error ~ normal(0, inv(quantity) * sigma_y);
}

generated quantities {
  matrix[periods_to_predict, D]             log_predicted_prices; 
  matrix[periods_to_predict, D]             delta_hat; // Trend at time t
  matrix[periods_to_predict, D]             tau_hat; // Seasonality at time t
  matrix[periods_to_predict, D]             omega_hat; // Cyclicality at time t
  matrix[periods_to_predict, D]             omega_star_hat; // Anti-cyclicality at time t
  matrix[periods_to_predict, D]             theta_hat; // Conditional variance of innovations 
  matrix[periods_to_predict, D]             epsilon_squared_hat; 
  
  // TREND
  for (t in 1:periods_to_predict) {
    row_vector[D] nu_hat = to_row_vector(multi_normal_cholesky_rng(to_vector(zero_vector), L_Omega_trend));
    if (t == 1) {
      delta_hat[t] = make_delta_t(alpha_trend, beta_trend, block(delta, P - ar, 1, ar, D), nu_hat);
    } else if (t <= ar) {
      delta_hat[t] = make_delta_t(alpha_trend, beta_trend, append_row(block(delta, P - ar + t, 1, ar - t - 1, D), 
                                                                      block(delta_hat, 1, 1, t-1, D)), nu_hat);
    } else {
      delta_hat[t] = make_delta_t(alpha_trend, beta_trend, block(delta_hat, periods_to_predict - ar, 1, ar, D), nu_hat);
    }
  }
  
  // SEASONALITY
  for (t in 1:(periods_to_predict)) {
    for (d in 1:D) {
      if (t <= s[d]) {
        tau_hat[t, d] = -sum(append_row(
          sub_col(tau_hat, 1, d, t - 1), 
          sub_col(tau, P - s[d] + t, d, s[d] - t + 1)
        )) + normal_rng(0, theta_season[d]);
      } else {
        tau_hat[t, d] = -sum(sub_col(tau_hat, t - 1 - s[d], d, s[d])) + normal_rng(0, theta_season[d]);
      }
    }
  }
  
  // Cyclicality
  {
    row_vector[D] rho_cos_lambda = rho .* cos(lambda); 
    row_vector[D] rho_sin_lambda = rho .* sin(lambda); 
    for (t in 1:(periods_to_predict)) {
      row_vector[D] kappa_hat = multi_normal_rng(zero_vector', diag_matrix(theta_cycle))';
      row_vector[D] kappa_star_hat = multi_normal_rng(zero_vector', diag_matrix(theta_cycle))';
      if (t == 1) {
        omega_hat[t] = (rho_cos_lambda .* omega[P-1]) + (rho_sin_lambda .* omega_star[P-1]) + kappa_hat;
        omega_star_hat[t] = -(rho_sin_lambda .* omega[P-1]) + (rho_cos_lambda .* omega_star[P-1]) + kappa_star_hat;
      } else {
        omega_hat[t] = (rho_cos_lambda .* omega_hat[t-1]) + (rho_sin_lambda .* omega_star_hat[t-1]) + kappa_hat;
        omega_star_hat[t] = -(rho_sin_lambda .* omega_hat[t-1]) + (rho_cos_lambda .* omega_star_hat[t-1]) + kappa_star_hat;   
      }
    }
  }
  
  
  // Univariate GARCH
  for (t in 1:periods_to_predict) {
    row_vector[D]  p_component; 
    row_vector[D]  q_component; 
    
    if (t <= p) {
      p_component = columns_dot_product(beta_p, append_row(
        block(theta, P - p + t, 1, p - t + 1, D),
        block(theta_hat, 1, 1, t - 1, D)
      ));
    } else {
      p_component = columns_dot_product(beta_p, block(theta_hat, t - p, 1, p, D));
    }
    
    if (t <= q) {
      q_component = columns_dot_product(beta_q, append_row(
        square(block(epsilon, P - q + t, 1, q - t + 1, D)),
        block(epsilon_squared_hat, 1, 1, t - 1, D)
      )); 
    } else {
      q_component = columns_dot_product(beta_q, block(epsilon_squared_hat, t - q, 1, q, D));
    }
    
    theta_hat[t] = omega_garch + p_component + q_component;
    epsilon_squared_hat[t] = square(multi_normal_cholesky_rng(zero_vector', make_L(theta_hat[t], L_omega_garch))');
  }
  
  
  {
    matrix[periods_to_predict, D] xi_hat = beta_xi * x_predictive;
    
    log_predicted_prices[1] = log_prices_hat[P] + delta_hat[1] + tau_hat[1] + omega_hat[1] + xi_hat[1] + sqrt(epsilon_squared_hat[1]);
    for (t in 2:(P-1)) {
      log_predicted_prices[t] = log_predicted_prices[t-1] + delta_hat[t] + tau_hat[t] + omega_hat[t] + xi_hat[t] + sqrt(epsilon_squared_hat[t]);
    }
  }
}

