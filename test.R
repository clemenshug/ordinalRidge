library(tidyverse)
library(qs)

## Objective function to minimize
## K - n-by-n kernel matrix
## y - n-by-1 vector of (ordinal) labels
## v - kernel weights
## b - vector of p bias terms, where p = #classes - 1
## lambda - ridge regularization coefficient
fobj <- function( K, y, v, b, lambda=1 ) {
    n <- nrow(K)
    p <- length(levels(y))-1
    
    ## Argument verification
    stopifnot( is.ordered(y) )
    stopifnot( length(b) == p )

    ## Convert true labels to indices
    j <- as.integer(y)-1
    
    ## Compute current fits for each threshold
    S <- t(t(matrix(K %*% v, n, p)) + b)

    ## Compute denominator terms
    DN <- log(1+exp(S))

    ## Threshold the fits according to the true label
    for( i in 1:ncol(S) ) S[,i] <- ifelse( j < i, 0, S[,i] )
    
    ## Compute log-likelihood
    LL <- sum(S - DN) / n

    ## Compute the ridge penalty
    R2 <- lambda * t(v) %*% K %*% v / 2

    ## Compute the ridge penalty and the overall objective
    drop( R2 - LL)
}

main <- function() {
    Z <- qread("traindata.qs")

    y <- pluck(Z, "Label")
    X <- select(Z, -Label)
    K <- X %>% t %>% cov

    ## Kernalize the bias term
    K1 <- rbind( K, 1 )
    K0 <- cbind( rbind( K, 0 ), 0 )

    n <- nrow(K)
    p <- length(levels(y))-1
    v <- rep(0, nrow(K))
    b <- rep(0, p)

    lambda <- 0.1
    eps <- 1e-5

    cat( "Initial objective function values: ", fobj(K, y, v, b, lambda), "\n" )

    for( iter in 1:10 ) {
        for( k in 1:p ) {
            ## Labels are defined by Y >= k
            yk <- as.numeric( (as.integer(y)-1) >=k )
    
            ## Compute the current fits
            s <- drop(K %*% v + b[k])
            pr <- 1 / (1 + exp(-s))

            ## Snap probabilities to 0 and 1 to avoid division by small numbers
            j0 <- which( pr < eps )
            j1 <- which( pr > (1-eps) )
            pr[j0] <- 0; pr[j1] <- 1

            ## Compute the sample weights and the response
            a <- pr * (1-pr)
            a[c(j0,j1)] <- eps
            z <- s + (yk - pr) / a

            ## Solve the ridge regression task using closed form
            A <- diag(a)
            mdl <- solve( K1 %*% A %*% t(K1) + lambda * n * K0, K1 %*% A %*% z )

            ## Update the weights
            v <- mdl[-(n+1)]
            b[k] <- mdl[n+1]
        }
        cat( "f = ", fobj(K, y, v, b, lambda), "after iteration", iter, "\n" )
    }

    ## Compute the final ranking of samples by the model
    ypred <- K %*% v

    ## Report correlation against true labels
    cat( "Final correlation =", cor(ypred, as.integer(y), method="kendall"), "\n" )
}
