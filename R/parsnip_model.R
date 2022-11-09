register_parsnip <- function() {
  if (!require("parsnip", quietly = TRUE))
    return()
  parsnip::set_new_model("oridge_rbf")
  parsnip::set_model_mode(model = "oridge_rbf", mode = "classification")
  parsnip::set_model_engine(
    "oridge_rbf",
    mode = "classification",
    eng = "oridge"
  )
  parsnip::set_dependency("oridge_rbf", eng = "oridge", pkg = "ordinalRidge")
  parsnip::set_model_arg(
    model = "oridge_rbf",
    eng = "oridge",
    parsnip = "penalty",
    original = "lambda",
    func = list(pkg = "dials", fun = "penalty"),
    has_submodel = FALSE
  )
  parsnip::set_model_arg(
    model = "oridge_rbf",
    eng = "oridge",
    parsnip = "rbf_sigma",
    original = "sigma",
    func = list(pkg = "dials", fun = "rbf_sigma"),
    has_submodel = FALSE
  )
  parsnip::set_model_arg(
    model = "oridge_rbf",
    eng = "oridge",
    parsnip = "kernel",
    original = "kernel",
    func = list(pkg = "foo", fun = "bar"),
    has_submodel = FALSE
  )
  parsnip::set_encoding(
    model = "oridge_rbf",
    eng = "oridge",
    mode = "classification",
    options = list(
      predictor_indicators = "none",
      compute_intercept = FALSE,
      remove_intercept = FALSE,
      allow_sparse_x = FALSE
    )
  )
  parsnip::set_fit(
    model = "oridge_rbf",
    eng = "oridge",
    mode = "classification",
    value = list(
      interface = "matrix",
      protect = c("x", "y"),
      func = c(pkg = "ordinalRidge", fun = "ordinalRidgeRBF"),
      defaults = list(eps = 1e-05, maxIter = 10, verbose = TRUE)
    )
  )
  # parsnip::set_pred(
  #   model = "oridge_rbf",
  #   eng = "oridge",
  #   mode = "classification",
  #   type = "numeric",
  #   value = list(
  #     pre = NULL,
  #     post = function(result, object) {
  #       result$score
  #     },
  #     func = c(fun = "predict"),
  #     args = list(
  #       mdl = quote(object$fit),
  #       newdata = quote(new_data)
  #     )
  #   )
  # )
  parsnip::set_pred(
    model = "oridge_rbf",
    eng = "oridge",
    mode = "classification",
    type = "prob",
    value = list(
      pre = NULL,
      post = function(result, object) {
        # browser()
        # p <- t(apply(result$prob, 1, function (x) c(1, x) - c(x, 0)))
        # colnames(p) <- levels(result$pred)
        tibble::as_tibble(result$prob)
      },
      func = c(fun = "predict"),
      args = list(
        mdl = quote(object$fit),
        newdata = quote(new_data)
      )
    )
  )
  parsnip::set_pred(
    model = "oridge_rbf",
    eng = "oridge",
    mode = "classification",
    type = "raw",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        mdl = quote(object$fit),
        newdata = quote(new_data)
      )
    )
  )
}

rbf_kernel <- function(x, y, sigma = 0.7) {
  d <- if (missing(y))
    dist(x)
  else
    pdist::pdist(x, y)
  exp(as.matrix(d)^2 / (-2 * sigma^2))
}

#' @export
oridge_rbf <- function(mode = "classification", penalty = NULL, rbf_sigma = NULL, kernel = c("rbf", "linear")) {
  if (mode != "classification") {
    rlang::abort("`mode` should be 'classification'")
  }
  args <- list(penalty = rlang::enquo(penalty), rbf_sigma = rlang::enquo(rbf_sigma), kernel = kernel)
  new_model_spec(
    "oridge_rbf",
    args = args,
    eng_args = NULL,
    mode = mode,
    method = NULL,
    engine = NULL
  )
}

#' @export
ordinalRidgeRBF <- function(x, y, kernel = c("rbf", "linear"), lambda = 0.1, sigma = 1, eps = 1e-05, maxIter = 10, verbose = TRUE) {
  kernel <- match.arg(kernel)
  if (kernel == "rbf") {
    rbf_mat <- rbf_kernel(x, sigma = sigma)
  } else {
    rbf_mat <- x
  }
  res <- ordinalRidge(
    rbf_mat, y,
    kernel = TRUE, lambda = lambda,
    eps = eps, maxIter = maxIter,
    verbose = verbose
  )
  res$sigma <- sigma
  res$training_data <- x
  res$kernel_type <- kernel
  class(res) <- c("ordinalRidgeRBF", class(res))
  pred <- predict(res, rbf_mat, no_probs = TRUE)
  polr_fit <- MASS::polr(
    y ~ score + 1, data = tibble(y = y, score = pred$score), Hess = TRUE
  )
  res$polr_fit <- polr_fit
  # message("original b ", paste(res$b, collase = " "))
  # message("zeta ", paste(polr_fit$zeta, collase = " "))
  # res$b <- rev(sort(-polr_fit$zeta/polr_fit$coefficients[1]))
  # message("new b ", paste(res$b, collase = " "))
  res
}

#' @export
predict.ordinalRidgeRBF <- function(mdl, newdata, no_probs = FALSE) {
  if (mdl$kernel_type == "rbf") {
    rbf_mat <- rbf_kernel(newdata, mdl$training_data, sigma = mdl$sigma)
  } else {
    rbf_mat <- newdata
  }
  pred <- predict_impl(mdl, rbf_mat)
  if (!no_probs) {
    class_probs <- predict(mdl$polr_fit, tibble(score = pred$score), type = "probs")
    message("calculated class probs")
    pred$prob <- class_probs
    # gte_probs <- 1 - apply(as.matrix(class_probs)[, -ncol(class_probs)], 1, cumsum)
    # pred$prob <- as_tibble(gte_probs)
  }
  pred
}

#' @export
ordinal_cross_entropy <- function(data, ...) {
  UseMethod("ordinal_cross_entropy")
}
ordinal_cross_entropy <- yardstick::new_prob_metric(
  ordinal_cross_entropy,
  direction = "minimize"
)

#' @export
ordinal_cross_entropy.data.frame <- function(
    data, truth, ..., na_rm = TRUE, event_level = "first", case_weights = NULL
) {
  estimate <- yardstick::dots_to_estimate(data, !!! enquos(...))
  yardstick::metric_summarizer(
    "ordinal_cross_entropy",
    ordinal_cross_entropy_vec,
    data = data,
    truth = !!rlang::enquo(truth),
    estimate = !!estimate,
    na_rm = na_rm,
    event_level = event_level,
    case_weights = !!enquo(case_weights)
  )
}

#' @export
ordinal_cross_entropy_vec <- function(
    truth, estimate, estimator = NULL, na_rm = TRUE,
    event_level = "first", case_weights = NULL,
    ...
) {
  estimator <- yardstick::finalize_estimator(truth, estimator)

  ordinal_cross_entropy_impl <- function(truth, estimate) {
    # browser()
    if (estimator == "binary") {
      yardstick:::mn_log_loss_binary(
        truth, estimate,
        event_level = "second", sum = FALSE,
        case_weights = case_weights
      )
    } else {
      gte_probs <- 1 - apply(estimate[, -ncol(estimate)], 1, cumsum)
      gte_prob_cols <- asplit(gte_probs, MARGIN = 1)
      losses <- purrr::imap_dbl(
        gte_prob_cols,
        function(col, pos) {
          yardstick:::mn_log_loss_binary(
            factor(as.integer(as.integer(truth) >= pos), levels = c(0, 1)), col,
            event_level = "second", sum = FALSE,
            case_weights = case_weights
          )
        }
      )
      message("losses ", paste(losses, collapse = " "))
      mean(losses)
    }
  }

  yardstick::metric_vec_template(
    metric_impl = ordinal_cross_entropy_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = c("ordered", "numeric"),
    estimator = estimator,
    ...
  )
}
