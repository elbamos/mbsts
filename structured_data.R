library(quantmod)

#
# Naive script which will evolve to something more sophisticated
# Generates multivariate time series with various structure
#

n.series <- 7
n.length <- 1000

t <- seq.Date(from=as.Date("1990-01-01"), by=1, length.out = n.length)
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
