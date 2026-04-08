library(glmnet)
library(randomForest)
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

# Very minimal cleaning that helped RMSE
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
              "distance", "hour", "dow", "passenger_count")

# Prepare a clean training dataframe with just the features we want
train_rf <- train_clean[, c(features, "fare_amount")]

rf_model <- randomForest(fare_amount ~ ., data = train_rf,
                         importance = TRUE,
                         mtry = 3,    
                         ntree = 500)

rf_model

# check if 500 trees is enough
plot(rf_model)

# Variable importance
importance(rf_model)

# Tuning mtry using 10-fold CV
ctrl <- trainControl(method = "cv", number = 10)

rf_grid <- expand.grid(mtry = 1:7)  # we have 8 features (became 16 later) so search 1 to 8

rf_cv_model <- train(fare_amount ~ ., data = train_rf,
                     method = "rf",
                     trControl = ctrl,
                     tuneGrid = rf_grid)

rf_cv_model          # show RMSE for each mtry
rf_cv_model$bestTune # best mtry

# RMSE on training set
rf_train_pred <- predict(rf_model, train_rf)
cat("RF RMSE (train):", sqrt(mean((train_rf$fare_amount - rf_train_pred)^2)), "\n")

# Predict on test set and write submission
test_rf  <- test_dat[, features]
rf_pred  <- predict(rf_model, test_rf)

#write to submission file
outDat = data.frame(uid = test_dat$uid, fare_amount = rf_pred)

write.csv(outDat, "W26P1_submission8.csv", row.names = F)


