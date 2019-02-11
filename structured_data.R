library(quantmod)
library(MASS)

# functions to generate data with semi-known properties

make.strtuctured.timeseries <- function(n.rows=1000, n.series=3, start=as.Date("1990-01-01")) {
  dt <- seq.Date(from=start, by=1, length.out = n.length)
  periods <- runif(n.series, 0, 0.05)
  diff.const <- runif(n.series, 0, 5)
  scale <- runif(n.series, 0, 20)
  intercept <- rnorm(n.series, 0, 10)
  
  data <- lapply(1:n.series, 
                 FUN = function(n) {
                   s <- intercept[n] +
                     scale[n]*sin(periods[n]*c(1:n.length)) + 
                     cumsum(rnorm(n.length, sd=diff.const[n]))
                   return(s)})
  X.structured <- xts(as.data.frame(data), order.by = t)
  colnames(X.structured) <- paste("S", c(1:n.series), sep=".")
  plot(X.structured, legend.loc = "topleft")
  return(X.structured)
}

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
  dates <- seq.Date(from=start, by=1, length.out = n.rows)
  X.xts <- xts(X, dates)
  return(X)
}

make.trend.data <- function(n.rows=1000, n.series=5, start=as.Date("1990-01-01")) {
  X <- mvrnorm(n=n.rows, mu=rep(0, n.series), Sigma=diag(n.series), empirical = TRUE)
  colnames(X) <- paste("X", 1:n.series, sep=".")
  dates <- seq.Date(from=start, by=1, length.out = n.rows)
  X.xts <- xts(X, dates)
  Z <- cumsum(X.xts)
}

make.ar.data <- function(n.rows=1000, n.series=3, start=as.Date("1990-01-01")) {
  X <- mvrnorm(n=n.rows, mu=rep(0, n.series), Sigma=diag(n.series), empirical = TRUE)
  colnames(X) <- paste("X", 1:n.series, sep=".")
  dates <- seq.Date(from=start, by=1, length.out = n.rows)
  X.xts <- xts(X, dates)
  Z <- rbind(X.xts[1,], cumsum(na.omit(X.xts * lag.xts(X.xts), -1)))
}

make.fake.timeseries <- function(n.rows=1000, n.series=3, start=as.Date("1990-01-01")) {
  X <- make.correlated.timeseries(n.rows, n.series, start)
  Y <- cumsum(X)
  Z <- make.ar.data(n.rows, n.series, start)
  return(Y + Z)
}

