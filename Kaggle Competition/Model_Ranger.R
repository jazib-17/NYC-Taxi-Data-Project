library(glmnet)
library(randomForest)
library(ranger)
library(caret)
library(doParallel) #install first
registerDoParallel(cores=16) #16 cores. It depends on how many cores your computer
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
  train_dat$fare_amount > 0 &
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

train_rf <- train_clean[, c(features, "fare_amount")]
test_rf  <- test_dat[, features]

ctrl <- trainControl(method = "cv", number = 10)

# splitrule = "variance" for regression ( "gini" is classification)
rf_grid <- expand.grid(mtry = 1:8,
                       splitrule = "variance",
                       min.node.size = 5)

rf_cv_model <- train(fare_amount ~ ., data = train_rf,
                     method = "ranger",
                     trControl = ctrl,
                     tuneGrid = rf_grid,
                     importance = 'impurity')

rf_cv_model

var_imp <- varImp(rf_cv_model)
print(var_imp)

# RMSE on training set
rf_train_pred <- predict(rf_cv_model, train_rf)
cat("RF RMSE (train):", sqrt(mean((train_rf$fare_amount - rf_train_pred)^2)), "\n")

# Final submission
rf_pred <- predict(rf_cv_model, test_rf)
outDat <- data.frame(uid = test_dat$uid, fare_amount = rf_pred)
write.csv(outDat, "W26P1_submission17.csv", row.names = FALSE)


