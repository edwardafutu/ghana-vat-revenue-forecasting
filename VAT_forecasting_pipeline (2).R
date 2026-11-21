# =====================================================================
#  PREDICTIVE MODELLING OF DOMESTIC VAT REVENUE IN GHANA
#  Baselines  : Naive, Seasonal Naive, ARIMAX (main baseline)
#  ML models  : LASSO, SVR, XGBoost, 1D-CNN
#
#  HOW TO USE
#  1. Put this file and full_dataset.xlsx in the SAME folder.
#  2. In RStudio: Session > Set Working Directory > To Source File Location
#  3. Run the script step by step (Ctrl+Enter) or all at once (Ctrl+Shift+Enter).
#  4. Pictures are saved in the folder  "figures"; tables in  "tables".
# =====================================================================


# ---------------------------------------------------------------------
# STEP 0 : GET READY (install and load the tools)
# ---------------------------------------------------------------------
# A "package" is a toolbox someone else built. We install each one once.

pkgs <- c('readxl', 'dplyr', 'tidyr', 'ggplot2', 'scales', 'tseries',
          'forecast', 'glmnet', 'e1071', 'xgboost', 'zoo', 'keras3')
missing_pkgs <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing_pkgs) > 0) install.packages(missing_pkgs)

# keras3 also needs Python/TensorFlow behind the scenes. Run this ONCE only,
# by hand, if you have never used keras before:
    keras3::install_keras()

suppressPackageStartupMessages({
  library(readxl);  library(dplyr);  library(tidyr);   library(ggplot2)
  library(scales);  library(tseries); library(forecast)
  library(glmnet);  library(e1071)
})
have_xgb   <- requireNamespace('xgboost', quietly = TRUE)
have_keras <- requireNamespace('keras3',  quietly = TRUE)
if (have_xgb)   library(xgboost)
if (have_keras) library(keras3)

set.seed(2026)                      # makes results repeatable
dir.create('figures', showWarnings = FALSE)
dir.create('tables',  showWarnings = FALSE)
theme_set(theme_minimal(base_size = 12))
save_plot <- function(p, name, w = 9, h = 5)
  ggsave(file.path('figures', name), p, width = w, height = h, dpi = 300, bg = 'white')


# ---------------------------------------------------------------------
# STEP 1 : LOAD THE DATA
# ---------------------------------------------------------------------
# The Excel column names are long and messy, so we rename them to short ones.

raw <- read_excel('full_dataset.xlsx', sheet = 'Sheet2')
names(raw) <- c('Date', 'Year', 'Month', 'VAT', 'CPI',
                'Inflation', 'FX', 'TBill', 'CIEA')
df <- raw %>% mutate(Date = as.Date(Date)) %>% arrange(Date)

glimpse(df)      # shows each column and its type
head(df)         # first 6 rows


# ---------------------------------------------------------------------
# STEP 2 : DATA CLEANING (check the data is healthy)
# ---------------------------------------------------------------------

# 2a. Missing values
cat('\nMissing values per column:\n'); print(colSums(is.na(df)))
# If you ever DO find missing values, fill them by drawing a straight line
# between the neighbouring months:
vars <- c('VAT', 'CPI', 'Inflation', 'FX', 'TBill', 'CIEA')
df <- df %>% mutate(across(all_of(vars), ~ zoo::na.approx(.x, na.rm = FALSE)))

# 2b. Duplicate months and gaps in the calendar
cat('Duplicate dates:', sum(duplicated(df$Date)), '\n')
all_months <- seq(min(df$Date), max(df$Date), by = 'month')
cat('No missing months:', length(all_months) == nrow(df) && all(all_months == df$Date), '\n')

# 2c. Impossible values (revenue or exchange rate should never be <= 0)
cat('Rows with VAT <= 0:', sum(df$VAT <= 0), '| FX <= 0:', sum(df$FX <= 0), '\n')

# 2d. Summary numbers
print(summary(df[, vars]))

# 2e. Outliers: look at the monthly % change in VAT.
#     A month is "unusual" if it is more than 3 standard deviations from normal.
df <- df %>% mutate(VAT_growth = (VAT / lag(VAT) - 1) * 100,
                    z = as.numeric(scale(VAT_growth)))
outliers <- df %>% filter(abs(z) > 3) %>% select(Date, VAT, VAT_growth, z)
cat('\nUnusual months (|z| > 3):\n'); print(outliers)
write.csv(outliers, 'tables/outlier_months.csv', row.names = FALSE)
# DECISION: we KEEP these months. They are probably real events (tax changes,
# COVID, reporting timing). Deleting them would hide what the models must learn.
# In your write-up, say you checked them and kept them.


# ---------------------------------------------------------------------
# STEP 3 : LOOK AT THE DATA (exploratory data analysis, with pictures)
# ---------------------------------------------------------------------

# 3a. VAT over time (normal scale)
p1 <- ggplot(df, aes(Date, VAT)) +
  geom_line(colour = '#1f4e79', linewidth = 0.8) +
  scale_y_continuous(labels = comma) +
  labs(title = 'Monthly Domestic VAT Revenue, 2010-2025',
       y = 'GHS million', x = NULL)
save_plot(p1, '01_vat_over_time.png'); print(p1)

# 3b. Same picture on a LOG scale (straight line = steady % growth)
p2 <- ggplot(df, aes(Date, VAT)) +
  geom_line(colour = '#1f4e79', linewidth = 0.8) +
  scale_y_log10(labels = comma) +
  labs(title = 'Domestic VAT on a log scale', y = 'GHS million (log)', x = NULL)
save_plot(p2, '02_vat_log_scale.png'); print(p2)

# 3c. All variables side by side
p3 <- df %>%
  select(Date, VAT, CPI, Inflation, FX, TBill, CIEA) %>%
  pivot_longer(-Date) %>%
  ggplot(aes(Date, value)) + geom_line(colour = '#2e75b6') +
  facet_wrap(~ name, scales = 'free_y', ncol = 2) +
  labs(title = 'All variables over time', x = NULL, y = NULL)
save_plot(p3, '03_all_variables.png', h = 7); print(p3)

# 3d. Seasonality: is some month always higher than others?
#     We divide each month by that year's average, so years are comparable.
seas <- df %>% group_by(Year) %>% mutate(ratio = VAT / mean(VAT)) %>% ungroup() %>%
  mutate(Mon = factor(Month, levels = 1:12, labels = month.abb))
p4 <- ggplot(seas, aes(Mon, ratio)) +
  geom_boxplot(fill = '#9dc3e6') +
  geom_hline(yintercept = 1, linetype = 'dashed') +
  labs(title = 'Seasonality: VAT relative to its yearly average',
       subtitle = 'Above the dashed line = stronger than average month',
       x = NULL, y = 'Ratio')
save_plot(p4, '04_seasonality.png'); print(p4)

# 3e. Split the series into Trend + Seasonal + Remainder
vat_ts <- ts(log(df$VAT), start = c(2010, 1), frequency = 12)
p5 <- autoplot(stl(vat_ts, s.window = 'periodic')) +
  labs(title = 'Decomposition of log(VAT)')
save_plot(p5, '05_decomposition.png', h = 7); print(p5)

# 3f. Monthly % change in VAT, with the unusual months marked in red
p6 <- ggplot(filter(df, !is.na(VAT_growth)), aes(Date, VAT_growth)) +
  geom_col(fill = 'grey70') +
  geom_point(data = outliers, aes(y = VAT_growth), colour = 'red', size = 2) +
  labs(title = 'Month-to-month % change in VAT (red = unusual months)',
       y = '% change', x = NULL)
save_plot(p6, '06_monthly_change_outliers.png'); print(p6)


# ---------------------------------------------------------------------
# STEP 4 : STATIONARITY CHECK  ("does the series keep drifting upward?")
# ---------------------------------------------------------------------
# Models struggle when a series just keeps climbing. We test with two tests:
#   ADF  : small p-value (< 0.05)  => series is steady (good)
#   KPSS : large p-value (> 0.05)  => series is steady (good)

check <- function(x, name) {
  x <- na.omit(x)
  data.frame(Series = name,
             ADF_p  = round(suppressWarnings(adf.test(x)$p.value), 3),
             KPSS_p = round(suppressWarnings(kpss.test(x, null = 'Level')$p.value), 3))
}
stat_table <- rbind(
  check(df$VAT,                       'VAT (raw level)'),
  check(log(df$VAT),                  'log(VAT)'),
  check(diff(log(df$VAT)),            'Change in log(VAT)  <- we use this'),
  check(diff(log(df$FX)),             'Change in log(FX)'),
  check(diff(df$TBill),               'Change in T-bill rate'),
  check(diff(log(df$CIEA)),           'Change in log(CIEA)')
)
print(stat_table)
write.csv(stat_table, 'tables/stationarity_tests.csv', row.names = FALSE)
# READ IT: the raw VAT fails both tests (it climbs forever). The monthly
# change passes. So we will predict the monthly CHANGE, not the raw amount.


# ---------------------------------------------------------------------
# STEP 5 : FEATURE ENGINEERING  (build the model's "clues")
# ---------------------------------------------------------------------
# TARGET = what we predict = monthly % change in VAT (log change).
#   dVAT_t = log(VAT_t) - log(VAT_t-1)
# Later we convert it back to GHS million, so the final error is in real money.
#
# CLUES = things known BEFORE the month starts (so we use LAGS = past values):
#   * VAT change 1, 2, 3 and 12 months ago  (12 months ago captures the yearly pattern)
#   * Last 1-2 months of: exchange-rate change, T-bill change,
#                         CIEA change (GDP proxy), monthly inflation
#   * Calendar: sin/cos of the month (smooth seasonal pattern) and a December flag
#
# Why not use this month's exchange rate etc.? Because in real life we would not
# know them yet when forecasting. Using them would be "cheating" (data leakage).

L <- dplyr::lag
df <- df %>% mutate(
  dVAT  = log(VAT)  - L(log(VAT)),
  dFX   = log(FX)   - L(log(FX)),
  dTB   = TBill     - L(TBill),
  dCIEA = log(CIEA) - L(log(CIEA)),
  Infl  = Inflation                       # already a monthly change
)

feat <- df %>% mutate(
  VAT_l1 = L(dVAT, 1), VAT_l2 = L(dVAT, 2), VAT_l3 = L(dVAT, 3), VAT_l12 = L(dVAT, 12),
  FX_l1  = L(dFX, 1),  FX_l2  = L(dFX, 2),
  TB_l1  = L(dTB, 1),  TB_l2  = L(dTB, 2),
  CIEA_l1 = L(dCIEA, 1), CIEA_l2 = L(dCIEA, 2),
  INF_l1 = L(Infl, 1), INF_l2 = L(Infl, 2),
  m_sin  = sin(2 * pi * Month / 12),
  m_cos  = cos(2 * pi * Month / 12),
  is_dec = as.numeric(Month == 12),
  VAT_prev   = L(VAT, 1),                 # last month's VAT (to convert back to GHS)
  VAT_prev12 = L(VAT, 12)                 # VAT a year ago (for seasonal-naive)
)

features <- c('VAT_l1', 'VAT_l2', 'VAT_l3', 'VAT_l12',
              'FX_l1', 'FX_l2', 'TB_l1', 'TB_l2', 'CIEA_l1', 'CIEA_l2',
              'INF_l1', 'INF_l2', 'm_sin', 'm_cos', 'is_dec')

model_df <- feat %>%
  select(Date, Year, Month, VAT, dVAT, VAT_prev, VAT_prev12, all_of(features)) %>%
  na.omit()                               # first 13 months have no 12-month lag
cat('\nRows available for modelling:', nrow(model_df), '\n')

# LAG CHECK: each row's "clues" must come from EARLIER months only.
# Look at the table: VAT_l1 in a row equals dVAT of the row ABOVE it.
lag_check <- feat %>% select(Date, dVAT, VAT_l1, dFX, FX_l1) %>% slice(15:19)
print(lag_check)

# Correlation picture of the target and the clues
cm <- cor(model_df[, c('dVAT', features)])
p7 <- as.data.frame(as.table(cm)) %>%
  ggplot(aes(Var1, Var2, fill = Freq)) + geom_tile() +
  geom_text(aes(label = round(Freq, 2)), size = 2.4) +
  scale_fill_gradient2(low = '#b2182b', mid = 'white', high = '#2166ac', limits = c(-1, 1)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = 'Correlation between the target (dVAT) and the clues',
       x = NULL, y = NULL, fill = 'Corr')
save_plot(p7, '07_correlation_heatmap.png', w = 9, h = 8); print(p7)


# ---------------------------------------------------------------------
# STEP 6 : SPLIT INTO TRAINING AND TEST DATA (respect time order!)
# ---------------------------------------------------------------------
# TRAIN = the past, used to teach the models        (Feb 2011 - Dec 2023)
# TEST  = the last 24 months, hidden until the end  (Jan 2024 - Dec 2025)
# We never shuffle. A model must never "see the future".

train <- model_df %>% filter(Date <= as.Date('2023-12-01'))
test  <- model_df %>% filter(Date >= as.Date('2024-01-01'))
cat('Train rows:', nrow(train), '| Test rows:', nrow(test), '\n')

p8 <- ggplot(model_df, aes(Date, VAT)) +
  annotate('rect', xmin = as.Date('2024-01-01'), xmax = max(model_df$Date),
           ymin = -Inf, ymax = Inf, fill = 'orange', alpha = 0.2) +
  geom_line(colour = '#1f4e79', linewidth = 0.8) +
  annotate('text', x = as.Date('2022-06-01'), y = max(model_df$VAT) * 0.95,
           label = 'Training data', colour = '#1f4e79') +
  annotate('text', x = as.Date('2025-01-01'), y = max(model_df$VAT) * 0.55,
           label = 'Test data', colour = 'darkorange3') +
  scale_y_continuous(labels = comma) +
  labs(title = 'Train / test split', y = 'GHS million', x = NULL)
save_plot(p8, '08_train_test_split.png'); print(p8)

x_tr <- as.matrix(train[, features]);  y_tr <- train$dVAT
x_te <- as.matrix(test[, features])

# TUNING without cheating: "rolling" validation.
# Fold 1: learn on everything before 2019, check on 2019.
# Fold 2: learn on everything before 2020, check on 2020 ... up to 2023.
folds <- lapply(2019:2023, function(y)
  list(train = which(train$Year < y), valid = which(train$Year == y)))
rmse <- function(a, p) sqrt(mean((a - p)^2))

# Helper: turn a predicted "% change" back into GHS million
to_level <- function(pred_d, prev_vat) prev_vat * exp(pred_d)

preds <- list()            # we collect every model's GHS predictions here


# ---------------------------------------------------------------------
# STEP 7 : BASELINE MODELS (the bar the ML models must beat)
# ---------------------------------------------------------------------

# 7a. NAIVE: "next month = this month"
preds[['Naive']] <- test$VAT_prev
# 7b. SEASONAL NAIVE: "this month = same month last year"
preds[['Seasonal Naive']] <- test$VAT_prev12

# 7c. ARIMAX: your main baseline.
#     = a classic ARIMA model (it learns from the past of VAT itself, including
#       the yearly seasonal pattern) PLUS extra inputs: the LAGGED macro variables
#       (exchange rate, T-bill, CIEA, inflation from the previous 1-2 months).
#     It predicts the same target as the ML models (monthly % change in VAT),
#     and uses the same lagged variables, so the comparison is fair.
#     auto.arima picks the best ARIMA settings by itself.
xreg_cols <- c('FX_l1', 'FX_l2', 'TB_l1', 'TB_l2',
               'CIEA_l1', 'CIEA_l2', 'INF_l1', 'INF_l2')
y_full    <- ts(model_df$dVAT, start = c(2011, 2), frequency = 12)
xreg_full <- as.matrix(model_df[, xreg_cols])
n_tr      <- nrow(train)

y_train   <- ts(model_df$dVAT[1:n_tr], start = c(2011, 2), frequency = 12)
fit_arimax <- auto.arima(y_train, xreg = xreg_full[1:n_tr, ],
                         seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
print(fit_arimax)

# Re-apply the SAME fitted model to the whole period. This gives one-step-ahead
# forecasts for each test month, using only information from earlier months.
refit <- Arima(y_full, xreg = xreg_full, model = fit_arimax)
pred_d_arimax <- as.numeric(fitted(refit))[(n_tr + 1):nrow(model_df)]
preds[['ARIMAX']] <- to_level(pred_d_arimax, test$VAT_prev)

# Check the ARIMAX leftovers (residuals): they should look like random noise.
png('figures/09_arimax_residuals.png', width = 2400, height = 1500, res = 300)
checkresiduals(fit_arimax)
invisible(dev.off())


# ---------------------------------------------------------------------
# STEP 8 : LASSO  (a regression that automatically drops useless clues)
# ---------------------------------------------------------------------
# lambda = how hard LASSO "punishes" clues. Big lambda => more clues dropped.
# We try many lambdas and keep the one with the smallest validation error.

lam_grid <- glmnet(x_tr, y_tr, alpha = 1)$lambda
cv_err <- sapply(folds, function(f) {
  fit <- glmnet(x_tr[f$train, ], y_tr[f$train], alpha = 1, lambda = lam_grid)
  p   <- predict(fit, x_tr[f$valid, ])
  sqrt(colMeans((y_tr[f$valid] - p)^2))
})
cv_mean     <- rowMeans(cv_err)
best_lambda <- lam_grid[which.min(cv_mean)]
cat('\nLASSO best lambda:', signif(best_lambda, 3), '\n')

lasso <- glmnet(x_tr, y_tr, alpha = 1, lambda = best_lambda)
preds[['LASSO']] <- to_level(as.numeric(predict(lasso, x_te)), test$VAT_prev)

# Picture 1: tuning curve
p10 <- data.frame(lambda = lam_grid, rmse = cv_mean) %>%
  ggplot(aes(log(lambda), rmse)) + geom_line(colour = '#1f4e79') +
  geom_vline(xintercept = log(best_lambda), linetype = 'dashed', colour = 'red') +
  labs(title = 'LASSO tuning: validation error vs lambda',
       subtitle = 'Red line = chosen lambda', x = 'log(lambda)', y = 'Validation RMSE')
save_plot(p10, '10_lasso_tuning.png'); print(p10)

# Picture 2: which clues did LASSO keep? (bar at 0 = clue was dropped)
cf <- coef(lasso)
coef_df <- data.frame(Feature = rownames(cf), Coef = as.numeric(cf)) %>%
  filter(Feature != '(Intercept)')
p11 <- ggplot(coef_df, aes(reorder(Feature, Coef), Coef, fill = Coef > 0)) +
  geom_col(show.legend = FALSE) + coord_flip() +
  scale_fill_manual(values = c('TRUE' = '#2e75b6', 'FALSE' = '#c55a11')) +
  labs(title = 'LASSO coefficients (clues kept by the model)',
       subtitle = 'Zero = removed by LASSO', x = NULL, y = 'Coefficient')
save_plot(p11, '11_lasso_coefficients.png'); print(p11)
write.csv(coef_df, 'tables/lasso_coefficients.csv', row.names = FALSE)


# ---------------------------------------------------------------------
# STEP 9 : SVR  (Support Vector Regression: fits a smooth, flexible curve)
# ---------------------------------------------------------------------
# Three dials to tune:
#   cost    = how strictly it tries to fit the training data
#   gamma   = how "wiggly" the curve may be
#   epsilon = size of the error it ignores
# We try every combination and keep the best one.

svr_grid <- expand.grid(cost = c(0.1, 1, 10, 100),
                        gamma = c(0.005, 0.02, 0.07, 0.2),
                        epsilon = c(0.01, 0.1))
svr_grid$cv_rmse <- sapply(seq_len(nrow(svr_grid)), function(i) {
  mean(sapply(folds, function(f) {
    m <- svm(x_tr[f$train, ], y_tr[f$train], type = 'eps-regression', kernel = 'radial',
             cost = svr_grid$cost[i], gamma = svr_grid$gamma[i],
             epsilon = svr_grid$epsilon[i], scale = TRUE)
    rmse(y_tr[f$valid], predict(m, x_tr[f$valid, ]))
  }))
})
best_svr <- svr_grid[which.min(svr_grid$cv_rmse), ]
print(best_svr)

svr <- svm(x_tr, y_tr, type = 'eps-regression', kernel = 'radial',
           cost = best_svr$cost, gamma = best_svr$gamma,
           epsilon = best_svr$epsilon, scale = TRUE)
preds[['SVR']] <- to_level(as.numeric(predict(svr, x_te)), test$VAT_prev)
write.csv(svr_grid[order(svr_grid$cv_rmse), ], 'tables/svr_tuning.csv', row.names = FALSE)

# Picture: tuning heat map
p12 <- ggplot(svr_grid, aes(factor(gamma), factor(cost), fill = cv_rmse)) +
  geom_tile() + facet_wrap(~ paste('epsilon =', epsilon)) +
  scale_fill_viridis_c(direction = -1) +
  labs(title = 'SVR tuning: validation error (darker = better)',
       x = 'gamma', y = 'cost', fill = 'RMSE')
save_plot(p12, '12_svr_tuning.png'); print(p12)


# ---------------------------------------------------------------------
# STEP 10 : XGBOOST  (many small decision trees, each fixing the last one's mistakes)
# ---------------------------------------------------------------------
# Dials: max_depth (tree size), eta (learning speed), nrounds (number of trees),
#        min_child_weight (stops tiny, noisy splits),
#        lambda (L2 penalty on leaf values), alpha (L1 penalty on leaf values),
#        gamma (minimum gain needed to make a split; higher = simpler trees).
# Adding 3 dials to a full grid would multiply the run time, so we RANDOMLY sample
# the same number of settings (24) as the old grid. Same number of model fits =
# same run time, but the search now also covers lambda, alpha and gamma.

if (have_xgb) {
  fit_xgb <- function(x, y, par) {
    set.seed(2026)
    xgb.train(params = list(objective = 'reg:squarederror',
                            max_depth = par$max_depth, eta = par$eta,
                            min_child_weight = par$min_child_weight,
                            lambda = par$lambda, alpha = par$alpha, gamma = par$gamma,
                            subsample = 0.8, colsample_bytree = 0.8),
              data = xgb.DMatrix(x, label = y), nrounds = par$nrounds, verbose = 0)
  }
  xgb_full <- expand.grid(max_depth = c(2, 3, 4), eta = c(0.03, 0.1),
                          nrounds = c(100, 300), min_child_weight = c(1, 5),
                          lambda = c(1, 5, 10), alpha = c(0, 0.1, 1),
                          gamma = c(0, 0.1, 1))
  set.seed(2026)                                   # same 24 settings every run
  xgb_grid <- xgb_full[sample(nrow(xgb_full), 24), ]   # 24 = size of the old grid
  rownames(xgb_grid) <- NULL
  xgb_grid$cv_rmse <- sapply(seq_len(nrow(xgb_grid)), function(i) {
    mean(sapply(folds, function(f) {
      m <- fit_xgb(x_tr[f$train, ], y_tr[f$train], xgb_grid[i, ])
      rmse(y_tr[f$valid], predict(m, x_tr[f$valid, ]))
    }))
  })
  best_xgb <- xgb_grid[which.min(xgb_grid$cv_rmse), ]
  print(best_xgb)

  xgb_final <- fit_xgb(x_tr, y_tr, best_xgb)
  preds[['XGBoost']] <- to_level(predict(xgb_final, x_te), test$VAT_prev)
  write.csv(xgb_grid[order(xgb_grid$cv_rmse), ], 'tables/xgboost_tuning.csv', row.names = FALSE)

  # Picture: which clues mattered most?
  imp <- xgb.importance(model = xgb_final)
  p13 <- ggplot(imp, aes(reorder(Feature, Gain), Gain)) +
    geom_col(fill = '#2e75b6') + coord_flip() +
    labs(title = 'XGBoost: importance of each clue', x = NULL, y = 'Importance (gain)')
  save_plot(p13, '13_xgboost_importance.png'); print(p13)
  write.csv(as.data.frame(imp), 'tables/xgboost_importance.csv', row.names = FALSE)
} else cat('\n>>> xgboost package not installed - XGBoost step skipped.\n')


# ---------------------------------------------------------------------
# STEP 11 : 1D-CNN  (a neural network that scans the last 12 months for patterns)
# ---------------------------------------------------------------------
# Instead of a table of clues, the CNN receives a "movie" of the past 12 months:
# for each month, 7 numbers (VAT change, inflation, FX change, T-bill change,
# CIEA change, month sin, month cos). Small "filters" slide along the 12 months
# looking for patterns (e.g. "two weak months then a jump").
# Because the data is small, we use a SMALL network and average 5 runs
# (neural networks are random, so averaging makes results more stable).

if (have_keras) tryCatch({
  Lw   <- 12
  chan <- c('dVAT', 'Infl', 'dFX', 'dTB', 'dCIEA', 'm_sin', 'm_cos')
  cnn_df <- df %>% mutate(m_sin = sin(2 * pi * Month / 12), m_cos = cos(2 * pi * Month / 12))

  # Scale every channel using TRAINING months only (mean 0, sd 1)
  tr_rows <- 2:which(cnn_df$Date == as.Date('2023-12-01'))
  mu <- sapply(cnn_df[tr_rows, chan], mean);  sdv <- sapply(cnn_df[tr_rows, chan], sd)
  Z  <- sweep(sweep(as.matrix(cnn_df[, chan]), 2, mu, '-'), 2, sdv, '/')

  make_windows <- function(dates) {
    rows <- match(dates, cnn_df$Date)
    X <- array(NA_real_, dim = c(length(rows), Lw, length(chan)))
    for (i in seq_along(rows)) X[i, , ] <- Z[(rows[i] - Lw):(rows[i] - 1), ]
    X
  }
  X_tr <- make_windows(train$Date);  X_te <- make_windows(test$Date)
  y_mu <- mean(y_tr);  y_sd <- sd(y_tr)
  y_tr_z <- (y_tr - y_mu) / y_sd

  l2_pen <- 1e-3     # L2 weight penalty (bigger = smaller weights = simpler network)
  build_cnn <- function() {
    keras_model_sequential(input_shape = c(Lw, length(chan))) %>%
      layer_conv_1d(filters = 16, kernel_size = 3, activation = 'relu',
                    kernel_regularizer = regularizer_l2(l2_pen)) %>%
      layer_conv_1d(filters = 16, kernel_size = 3, activation = 'relu',
                    kernel_regularizer = regularizer_l2(l2_pen)) %>%
      layer_flatten() %>%
      layer_dense(units = 16, activation = 'relu',
                  kernel_regularizer = regularizer_l2(l2_pen)) %>%
      layer_dropout(rate = 0.2) %>%
      layer_dense(units = 1) %>%
      compile(loss = 'mse', optimizer = optimizer_adam(learning_rate = 0.001))
  }

  cnn_preds <- NULL;  first_hist <- NULL
  for (s in 1:5) {
    set_random_seed(s)
    m <- build_cnn()
    h <- m %>% fit(X_tr, y_tr_z, epochs = 200, batch_size = 16,
                   validation_split = 0.15,       # last 15% of training = validation
                   callbacks = list(callback_early_stopping(monitor = 'val_loss',
                                                            patience = 20,
                                                            restore_best_weights = TRUE)),
                   verbose = 0)
    if (s == 1) first_hist <- h
    cnn_preds <- cbind(cnn_preds, as.numeric(predict(m, X_te, verbose = 0)))
  }
  pred_d <- rowMeans(cnn_preds) * y_sd + y_mu          # average, then un-scale
  preds[['1D-CNN']] <- to_level(pred_d, test$VAT_prev)

  # Picture: learning curve (should flatten, and the two lines should stay close)
  hist_df <- data.frame(epoch = seq_along(first_hist$metrics$loss),
                        Training = first_hist$metrics$loss,
                        Validation = first_hist$metrics$val_loss) %>%
    pivot_longer(-epoch)
  p14 <- ggplot(hist_df, aes(epoch, value, colour = name)) + geom_line(linewidth = 0.9) +
    labs(title = '1D-CNN learning curve', x = 'Epoch', y = 'Loss (MSE)', colour = NULL)
  save_plot(p14, '14_cnn_learning_curve.png'); print(p14)
}, error = function(e) cat('\n>>> 1D-CNN step skipped. Reason:', conditionMessage(e), '\n')
) else cat('\n>>> keras3 package not installed - 1D-CNN step skipped.\n')


# ---------------------------------------------------------------------
# STEP 12 : EVALUATION  (how wrong is each model, in GHS million?)
# ---------------------------------------------------------------------
#  RMSE : typical error, punishes big misses            (lower = better)
#  MAE  : average size of the error                     (lower = better)
#  MAPE : average error as a %  of the real value       (lower = better)

metrics <- function(actual, pred)
  c(RMSE = sqrt(mean((actual - pred)^2)),
    MAE  = mean(abs(actual - pred)),
    MAPE = mean(abs((actual - pred) / actual)) * 100)

results <- do.call(rbind, lapply(names(preds), function(m)
  data.frame(Model = m, t(round(metrics(test$VAT, preds[[m]]), 2)))))
naive_rmse <- results$RMSE[results$Model == 'Naive']
results$`RMSE vs Naive (%)` <- round((1 - results$RMSE / naive_rmse) * 100, 1)
results <- results[order(results$RMSE), ]
cat('\n================ TEST-SET RESULTS (Jan 2024 - Dec 2025) ================\n')
print(results, row.names = FALSE)
write.csv(results, 'tables/model_comparison.csv', row.names = FALSE)
cat('\nBest model by RMSE:', results$Model[1], '\n')

# Long table of predictions for plotting
plot_df <- data.frame(Date = test$Date, Actual = test$VAT, do.call(cbind, preds),
                      check.names = FALSE) %>%
  pivot_longer(-Date, names_to = 'Model', values_to = 'Value')
write.csv(data.frame(Date = test$Date, Actual = test$VAT, do.call(cbind, preds), check.names = FALSE),
          'tables/test_predictions.csv', row.names = FALSE)

# Picture 1: actual vs predicted over time
ml_models <- setdiff(names(preds), c('Naive', 'Seasonal Naive'))
p15 <- ggplot() +
  geom_line(data = filter(plot_df, Model %in% c('Actual', ml_models, 'ARIMAX')),
            aes(Date, Value, colour = Model, linewidth = Model == 'Actual')) +
  scale_linewidth_manual(values = c('TRUE' = 1.6, 'FALSE' = 0.8), guide = 'none') +
  scale_colour_manual(values = c('Actual' = 'black', 'ARIMAX' = 'grey50',
                                 'LASSO' = '#1f77b4', 'SVR' = '#2ca02c',
                                 'XGBoost' = '#d62728', '1D-CNN' = '#9467bd')) +
  scale_y_continuous(labels = comma) +
  labs(title = 'Actual vs predicted Domestic VAT (test period)',
       subtitle = 'One-step-ahead forecasts', y = 'GHS million', x = NULL)
save_plot(p15, '15_actual_vs_predicted.png', w = 10); print(p15)

# Picture 2: error metrics side by side
p16 <- results %>% select(Model, RMSE, MAE, MAPE) %>%
  pivot_longer(-Model, names_to = 'Metric') %>%
  ggplot(aes(reorder(Model, -value), value, fill = Model)) +
  geom_col(show.legend = FALSE) + coord_flip() +
  facet_wrap(~ Metric, scales = 'free_x') +
  labs(title = 'Model comparison on the test data (lower = better)', x = NULL, y = NULL)
save_plot(p16, '16_error_comparison.png', w = 11, h = 5); print(p16)

# Picture 3: predicted vs actual scatter (dots on the diagonal = perfect)
p17 <- filter(plot_df, Model != 'Actual') %>%
  left_join(select(plot_df %>% filter(Model == 'Actual'), Date, Actual = Value), by = 'Date') %>%
  ggplot(aes(Actual, Value)) + geom_point(colour = '#2e75b6') +
  geom_abline(linetype = 'dashed', colour = 'red') +
  facet_wrap(~ Model) + scale_x_continuous(labels = comma) + scale_y_continuous(labels = comma) +
  labs(title = 'Predicted vs actual (dashed = perfect)', x = 'Actual', y = 'Predicted')
save_plot(p17, '17_pred_vs_actual_scatter.png', w = 11, h = 7); print(p17)

# Picture 4: the mistakes over time (actual - predicted)
p18 <- filter(plot_df, Model != 'Actual') %>%
  left_join(select(plot_df %>% filter(Model == 'Actual'), Date, Actual = Value), by = 'Date') %>%
  mutate(Error = Actual - Value) %>%
  ggplot(aes(Date, Error)) + geom_col(fill = '#c55a11') + geom_hline(yintercept = 0) +
  facet_wrap(~ Model) +
  labs(title = 'Forecast errors over time (actual - predicted, GHS million)', x = NULL)
save_plot(p18, '18_errors_over_time.png', w = 11, h = 7); print(p18)

cat('\nDONE. Check the figures and tables folders.\n')
