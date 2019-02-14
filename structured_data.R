library(quantmod)
library(MASS)

# ------------------------------------------------------------
# functions to generate data with semi-known properties
# ------------------------------------------------------------

# this one has some periodicity, and some random walk
make.strtuctured.timeseries <- function(n.rows=1000, n.series=3, start=as.Date("1990-01-01")) {
  periods <- runif(n.series, 0, 0.1)
  diff.const <- runif(n.series, 0, 10)
  scale <- runif(n.series, 0, 50)
  intercept <- rnorm(n.series, 0, 50)
  
  data <- lapply(1:n.series, 
                 FUN = function(n) {
                   s <- intercept[n] +
                     scale[n]*sin(periods[n]*c(1:n.rows)) + 
                     cumsum(rnorm(n.rows, sd=diff.const[n]))
                   return(s)})
  dt <- seq.Date(from=start, by=1, length.out = n.rows)
  X.structured <- xts(as.data.frame(data), order.by = dt)
  colnames(X.structured) <- paste("S", c(1:n.series), sep=".")
  corr <- cor(X.structured)
  return(list(Z=X.structured, 
              params=list(cor=corr, periods=periods, diff.const=diff.const, intercept=intercept)))
}

# this will produce randomly cross-correlated time series
make.correlated.timeseries <- function(n.rows=1000, n.series=5, start=as.Date("1990-01-01")) {
  # make a correlation matrix
  n.cor <- n.series*(n.series - 1)/2
  corrs <- runif(n.cor, min = 0.01)*0.5
  R <- matrix(0, n.series, n.series)
  diag(R) <- rep(1, n.series)
  R[upper.tri(R)] <- corrs
  S <- t(R) %*% R
  X <- mvrnorm(n=n.rows, mu=rep(0, n.series), Sigma=S, empirical = TRUE)
  colnames(X) <- paste("X", 1:n.series, sep=".")
  corr <- cor(X)
  dates <- seq.Date(from=start, by=1, length.out = n.rows)
  Z.xts <- xts(X, dates)
  return(list(params=list(cor=corr), Z=Z.xts))
}

# this produces cumulatively summed random normal data
make.trend.data <- function(n.rows=1000, n.series=5, start=as.Date("1990-01-01")) {
  s <- runif(n.series, 0.1, 0.9)
  X <- mvrnorm(n=n.rows, mu=rep(0, n.series), Sigma=diag(s), empirical = TRUE)
  corr <- cor(X)
  colnames(X) <- paste("X", 1:n.series, sep=".")
  dates <- seq.Date(from=start, by=1, length.out = n.rows)
  X.xts <- xts(X, dates)
  Z <- cumsum(X.xts)
  return(list(params=list(cor=corr), Z=Z))
}

# produces a series whose innovations are Auto-regressive 
make.ar.data <- function(n.rows=1000, n.series=3, ar.order=1, start=as.Date("1990-01-01")) {
  s <- runif(n.series, 0.1, 0.9)
  X <- mvrnorm(n=n.rows + ar.order, mu=rep(0, n.series), Sigma=diag(s), empirical = TRUE)
  colnames(X) <- paste("X", 1:n.series, sep=".")
  dates <- seq.Date(from=start, by=1, length.out = nrow(X))
  X.xts <- xts(X, dates)
  phi <- runif(ar.order)
  for (i in c(1:ar.order)) {
    X.xts = X.xts + phi[i]*lag.xts(X.xts, -i)
  }
  Z.xts <- cumsum(na.omit(X.xts))
  corr <- cor(X)
  return(list(params=list(cor=corr, phi=phi), Z=Z.xts))
}

# put it all together - cross-correlated data with auto-regressive structure
make.fake.timeseries <- function(n.rows=1000, n.series=3, ar.order=1, start=as.Date("1990-01-01")) {
  X <- make.correlated.timeseries(n.rows, n.series, start)
  Y <- cumsum(X$Z)
  Z <- make.ar.data(n.rows, n.series, ar.order, start)
  corr <- cor(Z$Z)
  return(list(Z=Y + Z$Z, params=list(cor=corr, phi=Z$params$phi)))
}

# ------------------------------------------------------------
#  tests
# ------------------------------------------------------------
X.trend <- make.trend.data(n=1000, n.series=3)
X.correlated <- make.correlated.timeseries(n=1000, n.series=3)
X.autoreg <- make.ar.data(n=1000, n.series=3)
X.fake <- make.fake.timeseries(n=1000, n.series=3)
X.structured <- make.strtuctured.timeseries(n.rows=1000, n.series=3)

plot(X.trend$Z, legend.loc = "topleft")
print(X.trend$params)

plot(X.correlated$Z, legend.loc = "topleft")
print(X.correlated$params)

plot(X.autoreg$Z, legend.loc = "topleft")
print(X.autoreg$params)

plot(X.fake$Z, legend.loc = "topleft")
print(X.fake$params)

plot(X.structured$Z, legend.loc = "topleft")
print(X.structured$params)
