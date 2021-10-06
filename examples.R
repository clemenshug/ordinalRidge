library(tidyverse)
library(ordinalRidge)
library(pdist)
library(rsample)
library(here)

example_spiral <- function(n, n_classes = 5, spread = 0.5, angle = 4 * pi, noise_sd = 1) {
  tibble(
    phi = angle * (1 - cumsum(seq(0, 1, length.out = n))/sum(seq(0, 1, length.out = n))),
    x = sin(phi) * spread * phi,
    y = cos(phi) * spread * phi,
    class = cut(phi, breaks = n_classes, labels = as.character(seq_len(n_classes)), include.lowest = TRUE, ordered_result = TRUE)
  ) %>%
    arrange(phi) %>%
    mutate(across(c(x, y), ~.x + rnorm(length(.x), sd = noise_sd)))
}

example_zigzag <- function(n, n_classes = 5, random_spread = 0.5, spread = 1, amplitude = 3, phases = 3) {
  tibble(
    phi = seq(0, 2 * pi * phases, length.out = n),
    # phi = 2 * pi * phases * (1 - cumsum(seq(0, 1, length.out = n))/sum(seq(0, 1, length.out = n))),
    # y = {
    #   a = seq(0, phases * 2 * pi, length.out = n)
    #   s = cumsum(abs(sin(a))**0.2)
    #   phases * 2 * pi * s / s[length(s)]
    # },
    # y = {
    #   abs(sin(seq(0, phases * 2 * pi, length.out = n))) * seq(0, phases * 2 * pi, length.out = n)
    # },
    x = sin(phi) * amplitude,
    y = seq(0, 1, length.out = n) * spread,
    class = cut(y, breaks = n_classes, include.lowest = TRUE, ordered_result = TRUE)
  )
}

set.seed(42)
zigzag_df <- example_zigzag(phases = 1.5, 500)

ggplot(zigzag_df, aes(x, y, color = class)) +
  geom_point()

set.seed(42)
spiral_df <- example_spiral(
  1000, spread = 0.8, angle = 7 * pi, n_classes = 4
)

{ggplot(spiral_df, aes(x, y, color = class)) +
  geom_point()} %>%
  {ggsave(here("examples", "spiral_raw.pdf"), ., width = 4, height = 3)}

rbf_kernel <- function(x, y, sigma = 0.7) {
  d <- if (missing(y))
    dist(x)
  else
    pdist::pdist(x, y)
  exp(as.matrix(d)^2 / (-2 * sigma^2))
}

cum_to_single_probs <- function(df) {
  mat <- df %>%
    select(starts_with("Pr[")) %>%
    as.matrix()
  probs <- apply(
    mat, 1,
    function(x) c(1, x) - c(x, 0)
  ) %>%
    t() %>%
    as_tibble() %>%
    set_names(paste0("Pr[y == ", seq_len(ncol(.)), "]"))
  df %>%
    select(-starts_with("Pr[")) %>%
    bind_cols(probs)
}

crossval <- function(df, k = 10, sigma = 10, seed = 42) {
  set.seed(seed)
  samples <- rsample::vfold_cv(df, v = k)
  samples %>%
    mutate(
      res = map(
        splits,
        function(s) {
          train_mat <- training(s) %>%
            select(x, y) %>%
            as.matrix()
          test_mat <- testing(s) %>%
            select(x, y) %>%
            as.matrix()
          train_rbf <- rbf_kernel(train_mat, sigma = sigma)
          mod <- ordinalRidge(
            train_rbf, y = training(s)$class, kernel = TRUE
          )
          test_rbf <- rbf_kernel(
            test_mat, train_mat, sigma = 10
          )
          preds <- predict(mod, newdata = test_rbf)
          rank_score <- evaluateRanking(
            preds$prob[, ncol(preds$prob)], testing(s)[["class"]]
          )
          mse <- mean((as.integer(preds$pred) - as.integer(testing(s)$class))^2)
          list(rank_score = rank_score, mse = mse)
        }
      )
    )
}

crossval_conditions <- tibble(
  sigma = c(0.1, 0.5, 1, 2, 4, 8, 16, 32)
)
x <- crossval_conditions %>%
  mutate(
    scores = map(
      sigma,
      ~crossval(spiral_df, sigma = .x)
    )
  )

x_res <- x %>%
  unnest(scores) %>%
  unnest_wider(res) %>%
  mutate(mse = map_dbl(mse, mean))

ggplot(x_res, aes(mse, color = as.factor(sigma))) +
  geom_histogram() +
  facet_wrap(~as.factor(sigma))

spiral_mat <- spiral_df %>%
  select(x, y) %>%
  as.matrix()

spiral_rbf <- spiral_mat %>%
  rbf_kernel(sigma = 32)

# pheatmap::pheatmap(
#   spiral_rbf,
#   cluster_rows = FALSE, cluster_cols = FALSE,
#   color = colorRampPalette(RColorBrewer::brewer.pal(9, "Reds"), space = "Lab")(100),
#   labels_row = "", labels_col = ""
# )

withr::with_pdf(
  here("examples", "spiral_rbf_sigma32.pdf"),
  image(spiral_rbf, useRaster = TRUE), width = 5, height = 5
)


mod <- ordinalRidge(
  spiral_rbf, y = spiral_df$class, kernel = TRUE
)

{ggplot(
  tibble(v = mod$v, class = spiral_df$class), aes(x = v, color = class)
) +
  geom_density() #+ geom_vline(aes(xintercept = b), data = tibble(b = mod$b))
  } %>%
  {ggsave(here("examples", "spiral_v_sigma32.pdf"), ., width = 4, height = 2)}

preds_test <- predict(mod, newdata = spiral_rbf)

{ggplot(
  tibble(score = preds_test$score, class = spiral_df$class), aes(x = score, color = class)
) +
    geom_density() + geom_vline(aes(xintercept = b), data = tibble(b = mod$b))
} %>%
  {ggsave(here("examples", "spiral_score_sigma32.pdf"), ., width = 4, height = 2)}

preds_test_df <- preds_test$prob %>%
  as_tibble() %>%
  cum_to_single_probs %>%
  mutate(observation = seq_len(n())) %>%
  pivot_longer(cols = starts_with("Pr"), names_to = "pred_class", values_to = "prob") %>%
  arrange(observation, desc(prob)) %>%
  group_by(observation) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  bind_cols(spiral_df)
# preds_test_df <- spiral_df %>%
#   mutate(pred = preds_test$pred)

ggplot(preds_test_df, aes(x, y, color = pred_class)) +
  geom_point()

grid_coords <- expand_grid(
  x = seq(min(spiral_df$x), max(spiral_df$x), length.out = 100),
  y = seq(min(spiral_df$y), max(spiral_df$y), length.out = 100)
)

grid_rbf <- rbf_kernel(
  as.matrix(grid_coords), spiral_mat, sigma = 32
)

# pheatmap::pheatmap(
#   grid_rbf,
#   cluster_rows = FALSE, cluster_cols = FALSE,
#   color = colorRampPalette(RColorBrewer::brewer.pal(9, "Reds"), space = "Lab")(100),
#   labels_row = "", labels_col = ""
# )

image(grid_rbf, useRaster = TRUE)

preds <- predict(mod, newdata = grid_rbf)

pred_df <- preds$prob %>%
  as_tibble() %>%
  cum_to_single_probs() %>%
  bind_cols(grid_coords) %>%
  mutate(observation = seq_len(n()), score = preds$score) %>%
  pivot_longer(cols = -c(x, y, observation), names_to = "pred_class", values_to = "prob")

ggplot(filter(pred_df, pred_class == "score"), aes(x, y, fill = prob)) +
  geom_tile() +
  geom_point(aes(color = class, fill = NULL), data = spiral_df)

pred_df_max <- pred_df %>%
  arrange(observation, desc(prob)) %>%
  group_by(observation) %>%
  slice_head(n = 1) %>%
  ungroup()

ggplot(pred_df_max, aes(x, y, fill = pred_class)) +
  geom_tile()

ggplot(pred_df, aes(x, y, z = prob)) +
  geom_contour_filled() +
  facet_wrap(~pred_class)

ggplot(
  preds$prob %>%
    as_tibble() %>%
    bind_cols(grid_coords) %>%
    mutate(observation = seq_len(n())) %>%
    pivot_longer(cols = -c(x, y, observation), names_to = "pred_class", values_to = "prob"),
  aes(x, y, z = prob)
) +
  geom_contour_filled() +
  facet_wrap(~pred_class)

ggplot(spiral_df, aes(x, y, color = class)) +
  geom_density2d_filled(
    aes(group = pred_class, color = NULL),
    data = pred_df_max, contour_var = "ndensity"
  ) +
  geom_point() +
  facet_wrap(~pred_class) +
  scale_fill_viridis_d(option = "inferno")
  # scale_color_brewer(palette = "Set2")
  # scale_fill_distiller(palette = "Reds")



set.seed(42)
n <- 500
n_classes <- 3
circles_df <- tibble(
  # phi = seq(0, 4*pi, length.out = 100),
  phi = 4 * pi * seq(0, 1, length.out = n),
  x = 2.5 * rep(seq_len(n_classes), length.out = n) * sin(phi),
  y = 2.5 * rep(seq_len(n_classes), length.out = n) * cos(phi),
  class = ordered(rep(seq_len(n_classes), length.out = n))
) %>%
  arrange(class) %>%
  mutate(across(c(x, y), ~.x + rnorm(length(.x))))

ggplot(circles_df, aes(x, y, color = class)) +
  geom_point()

poly_kernel <- function(x, y = NULL, degree = 2, c = 0) {
  (tcrossprod(x, y) + c)^degree
}

circles_kernel <- circles_df %>%
  select(x, y) %>%
  as.matrix() %>%
  poly_kernel(c = 1) %>%
  magrittr::divide_by(2)

circles_kernel_2 <- kernlab::kernelMatrix(
  kernlab::polydot(degree = 2),
  circles_df %>%
    select(x, y) %>%
    as.matrix()
) /

pheatmap::pheatmap(
  circles_kernel,
  cluster_rows = FALSE, cluster_cols = FALSE,
  color = colorRampPalette(RColorBrewer::brewer.pal(9, "Reds"), space = "Lab")(100),
  labels_row = "", labels_col = ""
)

mod_circles <- ordinalRidge(
  circles_kernel, y = circles_df$class, kernel = TRUE
)

preds_circles <- predict(mod_circles, newdata = circles_kernel)

preds_circles_df <- preds_circles$prob %>%
  as_tibble() %>%
  cum_to_single_probs %>%
  mutate(observation = seq_len(n())) %>%
  pivot_longer(cols = starts_with("Pr"), names_to = "pred_class", values_to = "prob") %>%
  arrange(observation, desc(prob)) %>%
  group_by(observation) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  bind_cols(circles_df)
# preds_test_df <- spiral_df %>%
#   mutate(pred = preds_test$pred)

ggplot(preds_circles_df, aes(x, y, color = pred_class)) +
  geom_point()


grid_coords_circles <- expand_grid(
  x = seq(min(circles_df$x), max(circles_df$x), length.out = 100),
  y = seq(min(circles_df$y), max(circles_df$y), length.out = 100)
)

grid_circles_kernel <- poly_kernel(
  as.matrix(grid_coords_circles), as.matrix(select(circles_df, x, y)),
  c = 1
) / 2

preds_grid_circles <- predict(mod_circles, newdata = grid_circles_kernel)

pred_grid_circles_df <- preds_grid_circles$prob %>%
  as_tibble() %>%
  cum_to_single_probs() %>%
  bind_cols(grid_coords_circles) %>%
  mutate(observation = seq_len(n())) %>%
  pivot_longer(cols = -c(x, y, observation), names_to = "pred_class", values_to = "prob")

pred_grid_circles_df_max <- pred_grid_circles_df %>%
  arrange(observation, desc(prob)) %>%
  group_by(observation) %>%
  slice_head(n = 1) %>%
  ungroup()

ggplot(pred_df_max, aes(x, y, color = name)) +
  geom_point()

ggplot(pred_df, aes(x, y, z = value)) +
  geom_contour_filled() +
  facet_wrap(~name)

ggplot(
  preds_grid_circles$prob %>%
    as_tibble() %>%
    bind_cols(grid_coords_circles) %>%
    mutate(observation = seq_len(n())) %>%
    pivot_longer(cols = -c(x, y, observation), names_to = "pred_class", values_to = "prob"),
  aes(x, y, z = prob)
) +
  geom_contour_filled() +
  facet_wrap(~pred_class)

ggplot(circles_df, aes(x, y, color = class)) +
  geom_density2d_filled(
    aes(group = pred_class, color = NULL),
    data = pred_grid_circles_df_max, contour_var = "ndensity"
  ) +
  geom_point() +
  facet_wrap(~pred_class)


