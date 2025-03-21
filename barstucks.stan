data {
  int<lower=0> N; // number of observations
  int<lower=0> J; // number of stores
  int<lower=1, upper=J> store[N]; // store identifier for each observation
  vector[N] sales; // observed sales
  vector[N] con; // conscientiousness
  vector[N] food; // food indicator
  vector[N] neur; // neuroticism
}

parameters {
  real beta_0; // intercept
  real beta_con; // coefficient for conscientiousness
  real beta_food; // coefficient for food
  real beta_con_food; // interaction: con * food
  real beta_neur; // coefficient for neuroticism
  real beta_neur_food; // interaction: neur * food
  
  vector[2] u[J]; // random effects for each store
  real<lower=0> sigma; // residual standard deviation
  vector<lower=0>[2] tau; // standard deviations of random effects
  
  cholesky_factor_corr[2] L_Omega; // Cholesky factor of correlation matrix
}

transformed parameters {
  vector[N] mu;
  
  // Covariance matrix for random effects
  matrix[2, 2] L_Sigma;
  L_Sigma = diag_pre_multiply(tau, L_Omega);
  
  // Linear predictor
  for (n in 1:N) {
    mu[n] = beta_0 + beta_con * con[n] + beta_food * food[n] + beta_con_food * con[n] * food[n] +
            beta_neur * neur[n] + beta_neur_food * neur[n] * food[n] + 
            u[store[n], 1] + u[store[n], 2] * food[n];
  }
}

model {
  // Priors
  beta_0 ~ normal(0, 5);
  beta_con ~ normal(0, 5);
  beta_food ~ normal(0, 5);
  beta_con_food ~ normal(0, 5);
  beta_neur ~ normal(0, 5);
  beta_neur_food ~ normal(0, 5);

  // Random effects follow a multivariate normal distribution
  u ~ multi_normal_cholesky(rep_vector(0, 2), L_Sigma);

  L_Omega ~ lkj_corr_cholesky(3);
  sigma ~ cauchy(0, 2);
  tau ~ cauchy(0, 2);
  
  // Likelihood
  sales ~ normal(mu, sigma);
}
