library(glmnet)
library(gbm)
library(xgboost)
library(caret)
library(doParallel) #install first
registerDoParallel(cores=16) #16 cores.
set.seed(7)

#read data
train_dat <- read.csv("W26P1_train.csv", header = TRUE)
test_dat <- read.csv("W26P1_test.csv", header = TRUE)

# Haversine distance (in km)
haversine <- function(lat1, lon1, lat2, lon2) {
  R    <- 6371 # to get it in kilometres
  dlat <- (lat2 - lat1) * pi / 180
  dlon <- (lon2 - lon1) * pi / 180
  a    <- sin(dlat/2)^2 + cos(lat1*pi/180) * cos(lat2*pi/180) * sin(dlon/2)^2
  2 * R * asin(sqrt(a))
}

# Parse datetime for both sets
train_time <- as.POSIXct(as.character(train_dat$pickup_datetime))
test_time  <- as.POSIXct(as.character(test_dat$pickup_datetime))

# Distance
train_dat$distance <- haversine(train_dat$pickup_latitude, train_dat$pickup_longitude,
                                train_dat$dropoff_latitude, train_dat$dropoff_longitude)
test_dat$distance  <- haversine(test_dat$pickup_latitude,  test_dat$pickup_longitude,
                                test_dat$dropoff_latitude,  test_dat$dropoff_longitude)

# Hour of day (0–23)
train_dat$hour <- as.numeric(format(train_time, "%H"))
test_dat$hour  <- as.numeric(format(test_time,  "%H"))

# Day of week (1 = Monday … 7 = Sunday)
train_dat$dow  <- as.numeric(format(train_time, "%u"))
test_dat$dow   <- as.numeric(format(test_time,  "%u"))

#---------------------------------
# Addition of new features, sin_time,cos_time, airport pickups (JFK,LGA), 
# direction, city centre distance
#-----------------------------------
train_dat$sin_time <- sin(2 * pi * train_dat$hour / 24)
train_dat$cos_time <- cos(2 * pi * train_dat$hour / 24)

test_dat$sin_time <- sin(2 * pi * test_dat$hour / 24)
test_dat$cos_time <- cos(2 * pi * test_dat$hour / 24)

# I just searched LGA and JFK coordinates online
jfk_lat <- 40.645494; jfk_lon <- -73.785937
lga_lat <- 40.774071; lga_lon <- -73.872067

train_dat$pickup_dist_jfk  <- haversine(train_dat$pickup_latitude, train_dat$pickup_longitude, jfk_lat, jfk_lon)
train_dat$dropoff_dist_jfk <- haversine(train_dat$dropoff_latitude, train_dat$dropoff_longitude, jfk_lat, jfk_lon)

test_dat$pickup_dist_jfk  <- haversine(test_dat$pickup_latitude, test_dat$pickup_longitude, jfk_lat, jfk_lon)
test_dat$dropoff_dist_jfk <- haversine(test_dat$dropoff_latitude, test_dat$dropoff_longitude, jfk_lat, jfk_lon)

# LGA
train_dat$pickup_dist_lga  <- haversine(train_dat$pickup_latitude, train_dat$pickup_longitude, lga_lat, lga_lon)
train_dat$dropoff_dist_lga <- haversine(train_dat$dropoff_latitude, train_dat$dropoff_longitude, lga_lat, lga_lon)

test_dat$pickup_dist_lga  <- haversine(test_dat$pickup_latitude, test_dat$pickup_longitude, lga_lat, lga_lon)
test_dat$dropoff_dist_lga <- haversine(test_dat$dropoff_latitude, test_dat$dropoff_longitude, lga_lat, lga_lon)

train_dat$direction <- atan2(
  train_dat$dropoff_latitude - train_dat$pickup_latitude,
  train_dat$dropoff_longitude - train_dat$pickup_longitude
)

test_dat$direction <- atan2(
  test_dat$dropoff_latitude - test_dat$pickup_latitude,
  test_dat$dropoff_longitude - test_dat$pickup_longitude
)

# Seeing where most pickups were for getting a city centre idea
mean_lat <- mean(train_dat$pickup_latitude, na.rm = TRUE)
mean_lon <- mean(train_dat$pickup_longitude, na.rm = TRUE)

train_dat$pickup_dist_center <- haversine(
  train_dat$pickup_latitude,
  train_dat$pickup_longitude,
  mean_lat,
  mean_lon
)

test_dat$pickup_dist_center <- haversine(
  test_dat$pickup_latitude,
  test_dat$pickup_longitude,
  mean_lat,
  mean_lon
)

train_clean <- train_dat[
  train_dat$pickup_longitude > -75 & train_dat$pickup_longitude < -72 &
    train_dat$dropoff_longitude > -75 & train_dat$dropoff_longitude < -72 &
    train_dat$pickup_latitude > 40 & train_dat$pickup_latitude < 42 &
    train_dat$dropoff_latitude > 40 & train_dat$dropoff_latitude < 42,
]


# ── Prepare model matrices ─────────────────────────────────────────────────────

features <- c("pickup_longitude", "pickup_latitude",
              "dropoff_longitude", "dropoff_latitude",
              "distance", "hour", "dow", "passenger_count",
              "direction",
              "pickup_dist_jfk", "dropoff_dist_jfk",
              "pickup_dist_lga", "dropoff_dist_lga",
              "sin_time", "cos_time",
              "pickup_dist_center")

X_train <- as.matrix(train_clean[, features])
Y_train <- train_dat$fare_amount
X_test  <- as.matrix(test_dat[, features])

xgb_train <- xgb.DMatrix(data = X_train, label = Y_train)
xgb_test  <- xgb.DMatrix(data = X_test)

# Parameter defining
params <- list(
  booster   = "gbtree",
  eta       = 0.05,
  max_depth = 7,
  objective = "reg:squarederror",
  nthread   = 16
)

xgb_cv <- xgb.cv(
  params   = params,
  data     = xgb_train,
  nrounds  = 2000,
  nfold    = 10,
  verbose  = 1,
  early_stopping_rounds = 50
)

# What nrounds should I pick?
best_nrounds <- which.min(xgb_cv$evaluation_log$test_rmse_mean)
cat("Best nrounds:", best_nrounds, "\n")
cat("XGB CV RMSE:", min(xgb_cv$evaluation_log$test_rmse_mean), "\n")

xgb_fit <- xgb.train(params  = params,
                     data    = xgb_train,
                     nrounds = best_nrounds,
                     verbose = 1)

xgb_imp <- xgb.importance(feature_names = features, model = xgb_fit)
xgb_imp

xgb_train_pred <- predict(xgb_fit, xgb_train)
cat("XGB RMSE (train):", sqrt(mean((Y_train - xgb_train_pred)^2)), "\n")

# Write to submission file
xgb_pred <- predict(xgb_fit, xgb_test)
outDat <- data.frame(uid = test_dat$uid, fare_amount = xgb_pred)
write.csv(outDat, "W26P1_submission19.csv", row.names = FALSE)
