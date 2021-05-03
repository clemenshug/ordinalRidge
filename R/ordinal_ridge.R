#' Objective function to minimize
#'
#' @param K n-by-n kernel matrix
#' @param y n-by-1 vector of (ordinal) labels
#' @param v kernel weights
#' @param b vector of nb bias terms, where nb = #classes - 1
#' @param lambda ridge regularization coefficient
#' @return Objective function value
#' @export
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
#' @return OrdinalRidge returns a list with the following elements:
#' #' \describe{
#'   \item{v}{A n-by-1 vector of kernel weights}
#'   \item{b}{A nb-by-1 vector of bias terms to be used as decision boundaries}
#'   \item{classes}{Names of classes from the original vector of labels y}
#' }
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
        mdl <- MASS::ginv( K1 %*% (aa*K1t) + lambda * n * K0 ) %*% K1 %*% (aa*zz)

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

    structure( list( v=v, b=b, classes=levels(y) ), class="ordinalRidge" )
}

#' @param mdl model returned by `ordinalRidge()`
#' @param newdata n1-by-n kernel matrix of n1 new points against n points used for training
#' @describeIn ordinalRidge Predict method for ordinalRidge models
#' @return predict() returns a list with the following elements:
#' \describe{
#'   \item{score}{An n1-by-1 vector of scores}
#'   \item{pred}{A factor of length n1 containing predictions}
#'   \item{prob}{An n1-by-nb matrix of probabilities for each of the nb decision boundaries}
#' }
#' @export
predict.ordinalRidge <- function( mdl, newdata ) {
    predict_impl( mdl, newdata )
}

predict_impl <- function( mdl, newdata ) {
    res <- list()

    ## Scores
    res$score <- K %*% mdl$v

    ## Predictions
    res$pred <- factor( cut(res$score, breaks=c(-Inf,-mdl$b,Inf)), ordered=TRUE )
    levels(res$pred) <- res$classes

    ## Probabilities
    res$prob <- t(apply( res$score, 1, function(x) 1 / (1 + exp(-x - mdl$b)) ))
    colnames(res$prob) <- paste0("Pr[y >= ", 1:length(mdl$b), "]")

    res
}
