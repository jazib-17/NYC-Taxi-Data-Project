library(glmnet)
library(gbm)
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

# ── Prepare model matrices ─────────────────────────────────────────────────────

features <- c("pickup_longitude", "pickup_latitude",
              "dropoff_longitude", "dropoff_latitude",
              "distance", "hour", "dow", "passenger_count")

gbm_test  <- as.matrix(test_dat[, features])

# Prepare a clean training dataframe with just the features we want
train_rf <- train_dat[, c(features, "fare_amount")]

ctrl <- trainControl(method = "cv",
                     number = 10,
                     allowParallel = TRUE)

gbm_grid <- expand.grid(interaction.depth = c(5,6,7),
                        n.trees = (1:5)*200, 
                        shrinkage = c(0.1,0.05, 0.01),
                        n.minobsinnode    = 10)

gbm_cv_model <- train(fare_amount ~ ., data = train_rf,
                      method    = "gbm",
                      trControl = ctrl,
                      tuneGrid  = gbm_grid,
                      verbose   = FALSE)

gbm_cv_model
gbm_cv_model$bestTune

# RMSE on training set
gbm_train_pred <- predict(gbm_cv_model, train_rf)
cat("GBM RMSE (train):", sqrt(mean((train_rf$fare_amount - gbm_train_pred)^2)), "\n")

# Final submission
gbm_pred <- predict(gbm_cv_model, gbm_test)
outDat <- data.frame(uid = test_dat$uid, fare_amount = gbm_pred)
write.csv(outDat, "W26P1_submission10.csv", row.names = FALSE)

stopImplicitCluster()
