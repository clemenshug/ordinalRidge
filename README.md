# ordinalRidge

Fast solver for ordinal logistic regression with a ridge penalty

## Installation

The package can be installed directly from GitHub.

```r
if(!require(devtools)) install.packages("devtools")
devtools::install_github("ArtemSokolov/ordinalRidge")
```

## Example usage

Let's begin by generating training data. The package provides `toyData()`, which will generate points from sequential Gaussians.

``` r
library(ordinalRidge)
set.seed(42)

## Generate 3-D training data from three classes (2 decision boundaries), 100 points in each class
## Data for the first class will be sampled from N(0, 0.5)
## Data for the second class will be samples from N(1, 0.5)
## and so on...
## The three dimensions are assumed to be independent of each other
Tr <- toyData( 100, 3, 2, stdev=0.5 )

## Let's examine the first few entries
head(Tr$X)
#             Feat1       Feat2      Feat3
#  [1,]  0.68547922  0.60048269 -1.0004646
#  [2,] -0.28234909  0.52237554  0.1668886
#  [3,]  0.18156421 -0.50160432  0.5856626
#  [4,]  0.31643130  0.92424095  1.0297696
#  [5,]  0.20213416 -0.33338670 -0.6884308
#  [6,] -0.05306226  0.05275691 -0.5754278

head(Tr$y)
#  [1] 0 0 0 0 0 0
#  Levels: 0 < 1 < 2
```

The package provides a kernel implementation of ordinal regression. The [kernel trick](https://en.wikipedia.org/wiki/Kernel_method#Mathematics:_the_kernel_trick) allows the method to learn non-linear decision boundaries when supplied with, e.g., a polynomial or a Gaussian kernel. When using a simple linear kernel, the method will learn linear decision boundaries in the original feature space.

``` r
## Compute the 300-by-300 linear kernel matrix on training data
K <- Tr$X %*% t(Tr$X)

## Train an ordinal regression model
mdl <- ordinalRidge(K, Tr$y, verbose=FALSE)

## The weights for the original features can be computed from the kernel weights
t(Tr$X) %*% mdl$v
#             [,1]
#  Feat1 1.0166420
#  Feat2 0.9938466
#  Feat3 1.0025878

## The decision boundaries are given by the model bias terms
mdl$b
# [1] -1.393522 -4.517127
```

New data can be classified with two simple steps: 1) Computing the kernel matrix capturing relationships between test data points and training data, and 2) using `predict()` function to compute scores, probabilities and predictions.

``` r
## Generate 5 new points in each class
Te <- toyData( 5, 3, 2, stdev=0.5 )

## Compute a linear kernel between training and test points
K1 <- Te$X %*% t(Tr$X)
dim(K1)
#  [1]  15 300

## Compute predictions on the new 15 points
res <- predict(mdl, K1)
#  $score
#   [1] -0.883101093  0.007894235  0.734711848  0.147927491  0.892516226
#   [6]  3.734677963  2.171482370  2.183887040  2.685519749  3.414103877
#  [11]  4.491177062  6.182563822  7.438917673  5.283894638  4.435537524
# 
#  $pred
#   [1] 0 0 0 0 0 1 1 1 1 1 1 2 2 2 1
#  Levels: 0 < 1 < 2
# 
#  $prob
#        Pr[y >= 1]  Pr[y >= 2]
#   [1,] 0.09307759 0.004495251
#   [2,] 0.20010662 0.010887066
#   [3,] 0.34100687 0.022260806
#   [4,] 0.22346362 0.012503062
#   [5,] 0.37730425 0.025967193
#   [6,] 0.91222866 0.313792243
#   [7,] 0.68524028 0.087412561
#   [8,] 0.68790963 0.088407180
#   [9,] 0.78448508 0.138046882
#  [10,] 0.88294113 0.249173829
#  [11,] 0.95679590 0.493512810
#  [12,] 0.99174823 0.840966445
#  [13,] 0.99763685 0.948913162
#  [14,] 0.97997160 0.682821194
#  [15,] 0.95443654 0.479613870
```

We observe that points 11 and 15 get misclassified; the true labels for both are `2`, but they get misclassified as `1`. Looking at the `$score` output, we can see that both points are very close to the decision boundary (which is defined by `mdl$b[2]` or `-4.517127`) and happen to land on the negative side of it:
``` r
res$score[11:15] + mdl$b[2]
#  [1] -0.02595022  1.66543655  2.92179040  0.76676736 -0.08158975
```
This is also characterized by `Pr[y >= 2]` being just under 0.5 for both points. Note that even though both points are misclassified, they still receive higher `score` values that all points in classes `0` and `1`, indicating that the model nevertheless correctly ranks all test points.
