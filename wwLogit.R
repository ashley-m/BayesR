library(brms)
library(tidyverse)
ww_tr = read.csv("whitewine-training-ds6040-1.csv")
ww_ts = read.csv("whitewine-testing-ds6040.csv")
ww_tr <- ww_tr %>%
  mutate(wine_quality = ifelse(wine_quality =='A', 1, 0))
ww_ts <- ww_ts %>%
  mutate(wine_quality = ifelse(wine_quality =='A', 1, 0))

priors <- c(
  set_prior("normal(0, 5)", class = "b"),
  set_prior("normal(0, 5)", class = "Intercept")
)
f_mod_s = brm(wine_quality ~ residual.sugar + density + pH, data = ww_tr, prior = priors,
            family = bernoulli(), cores = 12, seed = 666)

priors <- c(
  set_prior("normal(0, 2.5)", class = "b"),
  set_prior("normal(1, 5)", class = "Intercept")
)
f_a_mod_s = brm(wine_quality ~ chlorides + volatile.acidity + sulphates, data = ww_tr[ww_tr$wine_quality==1, ], prior = priors,
              family = bernoulli(), cores = 12, seed = 666)
