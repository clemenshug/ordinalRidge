#' Generates toy data from sequential Gaussian distributions
#'
#' The n[i] data points for the i^th class are generated from N(i,stdev[i]).
#' Coordinates are assumed to be indepedent (i.e., covariance is the identity matrix).
#' 
#' @param n     Desired number of samples in each class. Values are recycled.
#' @param p     Data dimensionality
#' @param nb    Number of boundaries (== number of classes - 1)
#' @param stdev Standard deviation of each Gaussian. Values are recycled.
#' @return A list with the following elements:
#' \describe{
#'   \item{X}{A ns-by-p matrix of ns = sum(n) data points in p dimensions}
#'   \item{y}{A matching ordered factor of length ns containing true labels}
#' }
#' @export
toyData <- function( n, p, nb=6, stdev=1 ) {
    
    ## Recycle values to ensure there is one value per cluster
    n     <- rep_len(n, nb+1)
    stdev <- rep_len(stdev, nb+1)

    ## Generate data for each class
    X <- do.call(
        rbind,
        lapply( 0:nb, function(i)
            matrix( rnorm(n[i+1]*p, i, stdev[i+1]), n[i+1], p ) )
    )

    ## Name features
    colnames(X) <- paste0("Feat", 1:p)

    ## Compose the matching set of true labels
    y <- factor( rep(0:nb, n), levels=0:nb, ordered=TRUE )

    list(X=X,y=y)
}
