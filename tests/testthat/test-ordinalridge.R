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
