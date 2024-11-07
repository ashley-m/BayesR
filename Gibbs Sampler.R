library(Rmpfr)
dat = read.csv("coaldisasters-ds6040.csv")


gibbs_sampler = function(iter, dat, a_mu, b_mu, a_lambda, b_lambda){
  
  mu_vec = vector()
  lambda_vec = vector() 
  k_prob_mat = matrix(nrow = iter+1, ncol = 111)
  k_samp_vec = vector()
  #Initialize sampler
  mu_vec[1] = rgamma(1,a_mu, rate  = b_mu)
  lambda_vec[1] = rgamma(1,a_lambda, rate = b_lambda)
  k_prob_mat[1,] = rep(1/111, 111)
  k_samp_vec[1] = 56
  
  #Sampler
  for(i in 2:(iter+1)){
    mu_vec[i] = rgamma(1, a_mu + sum(dat[1:k_samp_vec[i - 1], 2]), rate = b_mu + k_samp_vec[i - 1])
    lambda_vec[i] = rgamma(1, a_lambda + sum(dat[(k_samp_vec[i - 1] + 1):112, 2]), rate = b_lambda + (112 - k_samp_vec[i - 1]))
    
    
    l_temp = vector()
    for(j in 1:111){  
      l_temp[j] = sum(log(mpfr(dpois(dat[1:j,2], lambda = rep(mu_vec[i],j)), precBits = 100))) + sum(log(mpfr(dpois(dat[(j+1):112,2], lambda = rep(lambda_vec[i],112-j)), precBits = 100)))
    }
    l_temp <- mpfr(l_temp, precBits = 100)
    k_prob_mat[i,] = as.numeric(exp(l_temp)/sum(exp(l_temp))) 
    k_samp_vec[i] = sample(size = 1,1:111, prob = k_prob_mat[i,])
  }
  toReturn = data.frame(mu = mu_vec, lambda = lambda_vec, k = k_samp_vec)
  
  return(toReturn)
}

test = gibbs_sampler(1000, dat, a_mu = 1, b_mu = 1, a_lambda = 1, b_lambda = 1)