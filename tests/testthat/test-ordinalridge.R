test_that("fitting works", {
  data( ad_data )
  Y <- ad_data$Label
  XX <- cov( t( ad_data[, names( ad_data ) != "Label"] )  )
  mdl <- ordinalRidge( XX, Y )
  expect_length( mdl$b, 6 )
  expect_length( mdl$v, 259 )
})

test_that("predicting works", {
  data( ad_data )
  Y <- ad_data$Label
  XX <- cov( t( ad_data[, names( ad_data ) != "Label"] )  )
  mdl <- ordinalRidge( XX, Y )
  pred <- predict( mdl, XX )
  expect_length( pred$class, 259 )
  expect_equal( dim( pred$p ), c(7, 259) )
})

test_that("ranking evaluation is accurate", {
  lbl <- c( 1, 1, 1, 2, 2, 2, 3, 3, 3 )
  scores <- c( 0.1, 0.2, 1.5, 1.2, 1.1, 1.3, 1.8, 2.5, 0.3 )
  expect_equal( evaluate_ranking(lbl, scores), 0.74074074 )
})
