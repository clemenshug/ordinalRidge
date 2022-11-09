test_that("fitting works", {
  set.seed(42)
  Tr <- toyData( 5, 3, 2, stdev=0.5 )
  XX <- cov( t(Tr$X) )
  mdl <- ordinalRidge( XX, Tr$y )
  expect_length( mdl$b, 2 )
  expect_length( mdl$v, 15 )
})

test_that("predicting works", {
  set.seed(42)
  Tr <- toyData( 5, 3, 2, stdev=0.5 )
  XX <- cov( t(Tr$X) )
  mdl <- ordinalRidge( XX, Tr$y )
  pred <- predict( mdl, XX )
  expect_length( pred$pred, 15 )
  expect_equal( dim( pred$prob ), c(15, 2) )
})

test_that("ranking evaluation is accurate", {
  lbl <- c( 1, 1, 1, 2, 2, 2, 3, 3, 3 )
  scores <- c( 0.1, 0.2, 1.5, 1.2, 1.1, 1.3, 1.8, 2.5, 0.3 )
  expect_equal( evaluateRanking(scores, lbl), 0.74074074 )
})

test_that("estimates for matrices with large values are stable", {
  ## Generate data and compute a linear kernel
  set.seed(42)
  Tr <- toyData( 100, 3, 3, stdev=2 )
  K <- Tr$X %*% t(Tr$X)

  ## Train a model using the original data and the kernel
  mdl1 <- ordinalRidge( Tr$X, Tr$y )
  mdl2 <- ordinalRidge( K, Tr$y )

  ## Ensure that both models produce the same predictions
  pred1 <- predict( mdl1, Tr$X )
  pred2 <- predict( mdl2, K )
  expect_equal(range(pred1$score - pred2$score), c(0, 0))
})

test_that("parsnip interface works", {
  skip_if_not_installed("parsnip")
  ## Generate data and compute a linear kernel
  set.seed(42)
  Tr <- toyData( 100, 3, 3, stdev=2 )

  K <- Tr$X %*% t(Tr$X)

  mdl_spec <- oridge_rbf() %>%
    parsnip::set_engine("oridge")
  mdl_fit <- mdl_spec %>%
    fit_xy(x = Tr$X, y = Tr$y)

  Te <- toyData( 100, 3, 3, stdev=2 )
  pred <- predict(mdl_fit, new_data = Te$X, type = "prob")
  pred <- predict(mdl_fit, new_data = Te$X, type = "raw")

  ## Train a model using the original data and the kernel
  mdl1 <- ordinalRidge( Tr$X, Tr$y )
  mdl2 <- ordinalRidge( K, Tr$y )

  ## Ensure that both models produce the same predictions
  pred1 <- predict( mdl1, Tr$X )
  pred2 <- predict( mdl2, K )
  expect_equal(range(pred1$score - pred2$score), c(0, 0))
})
