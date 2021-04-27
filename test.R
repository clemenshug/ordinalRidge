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

    ## Compute the overall objective value
    drop(R2 - LL)
}

## Ordinal regression with a ridge regularization penalty
## K - n-by-n kernel matrix
## y - n-by-1 vector of (ordinal) labels
## lambda - regularization coefficient
## eps - convergence tolerance
## maxIter - maximum number of iterations
ordinalRidge <- function( K, y, lambda=0.1, eps=1e-5, maxIter=10 ) {
    ## Determine the problem dimensions
    n <- nrow(K)
    p <- length(levels(y))-1

    ## Kernalize the bias terms
    K1 <- rbind( matrix(1,1,p) %x% K, diag(p) %x% matrix(1,1,n) )
    K0 <- cbind( rbind(K, matrix(0,p,n)), matrix(0,n+p,p) )

    ## Initial parameter estimates
    v <- rep(0, nrow(K))
    b <- rep(0, p)

    fprev <- fobj(K, y, v, b, lambda)
    cat( "Initial f = ", fprev, "\n" )

    for( iter in 1:maxIter ) {

        ## Build up residuals and weights across decision boundaries
        aa <- c()
        zz <- c()
        
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

            ## Contribute to the joint problem
            aa <- c(aa, a)
            zz <- c(zz, z)
        }
            
        ## Solve the ridge regression task using closed form
        A <- diag(aa)
        mdl <- solve( K1 %*% A %*% t(K1) + lambda * n * K0, K1 %*% A %*% zz )

        ## Update the weights
        v <- mdl[1:n]
        b <- mdl[(n+1):(n+p)]

        f <- fobj(K, y, v, b, lambda)
        if( abs(f - fprev) / abs(fprev) < eps ) break
        else {
            cat( "f = ", f, "after iteration", iter, "\n" )
            fprev <- f
        }
    }
    cat( "Final f = ", f, "after iteration", iter, "\n" )

    list( v=v, b=b )
}

main <- function() {
    Z <- qs::qread("traindata.qs")

    y <- purrr::pluck(Z, "Label")
    X <- as.matrix( dplyr::select(Z, -Label) )
    K <- cov(t(X))

    mdl <- ordinalRidge( K, y )

    ## Compute the final ranking of samples by the model
    ypred <- K %*% mdl$v

    ## Report correlation against true labels
    cat( "\n" )
    cat( "Correlation against ground truth =", cor(ypred, as.integer(y), method="kendall"), "\n" )    
}
