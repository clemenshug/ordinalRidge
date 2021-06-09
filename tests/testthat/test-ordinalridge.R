test_that("fitting works", {
  Tr <- toyData( 5, 3, 2, stdev=0.5 )
  XX <- cov( t(Tr$X) )
  mdl <- ordinalRidge( XX, Tr$y )
  expect_length( mdl$b, 2 )
  expect_length( mdl$v, 15 )
})

test_that("predicting works", {
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
