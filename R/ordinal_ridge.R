#' Objective function to minimize
#'
#' @param K n-by-n kernel matrix
#' @param y n-by-1 vector of (ordinal) labels
#' @param v kernel weights
#' @param b vector of p bias terms, where p = #classes - 1
#' @param lambda ridge regularization coefficient
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

#' Ordinal regression with a ridge regularization penalty
#'
#' @param K n-by-n kernel matrix
#' @param y n-by-1 vector of (ordinal) labels
#' @param lambda regularization coefficient
#' @param eps convergence tolerance
#' @param maxIter maximum number of iterations
#' @param verbose if TRUE, reports objective function value at each iteration
#' @export
ordinalRidge <- function( K, y, lambda=0.1, eps=1e-5, maxIter=10,
                         verbose=TRUE ) {
    ## Determine the problem dimensions
    n <- nrow(K)
    p <- length(levels(y))-1

    ## Kernalize the bias terms
    K1 <- rbind( matrix(1,1,p) %x% K, diag(p) %x% matrix(1,1,n) )
    K1t <- t(K1)
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
        Kv <- K %*% v

        for( k in 1:p ) {
            ## Labels are defined by Y >= k
            yk <- as.numeric( (as.integer(y)-1) >=k )

            ## Compute the current fits
            s <- drop(Kv + b[k])
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
        mdl <- solve( K1 %*% (aa*K1t) + lambda * n * K0, K1 %*% (aa*zz) )

        ## Update the weights
        v <- mdl[1:n]
        b <- mdl[(n+1):(n+p)]

        f <- fobj(K, y, v, b, lambda)
        if( abs(f - fprev) / abs(fprev) < eps ) break
        else {
            fprev <- f
            if( verbose ) cat( "f = ", f, "after iteration", iter, "\n" )
        }
    }
    cat( "Final f = ", f, "after iteration", iter, "\n" )

    structure( predict_impl( list( v=v, b=b ), K ), class="ordinalRidge" )
}

#' @param mdl model returned by `ordinalRidge()`
#' @param newdata n-by-n kernel matrix to run predictions on
#' @describeIn ordinalRidge Predict method for ordinalRidge models
#' @export
predict.ordinalRidge <- function( mdl, newdata ) {
    predict_impl( mdl, newdata )
}

predict_impl <- function( mdl, newdata ) {
    mdl$p <- NULL
    mdl$class <- NULL
    s <- newdata %*% mdl$v
    p_cum <- apply( s, 1, function (x) 1 / (1 + exp(-x - mdl$b)) )
    p <- apply( p_cum, 2, function (x) c(1, x) - c(x, 0) )
    class <- apply( p, 2, which.max )
    c( mdl, list( p=p, class=class ) )
}
