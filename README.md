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

The package provides a kernel implementation of ordinal regression. The [kernel trick](https://en.wikipedia.org/wiki/Kernel_method#Mathematics:_the_kernel_trick) allows the method to learn non-linear decision boundaries when supplied with, e.g., a polynomial or a Gaussian kernel.
