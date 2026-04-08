library(glmnet)
library(mgcv)
library(caret)
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

X_train <- as.matrix(train_dat[, features])
Y_train <- train_dat$fare_amount
X_test  <- as.matrix(test_dat[, features])

train_clean <- train_dat[!(train_dat$distance < 0.5 & train_dat$fare_amount > 10), ]

train_rf <- train_clean[, c(features, "fare_amount")]

# Fit GAM — smooth terms for continuous features, linear for discrete
gam_model <- gam(fare_amount ~ s(distance, bs = "cr") +
                   s(hour, bs = "cr") +
                   s(pickup_longitude, bs = "cr") +
                   s(pickup_latitude, bs = "cr") +
                   s(dropoff_longitude, bs = "cr") +
                   s(dropoff_latitude, bs = "cr") +
                   dow +
                   passenger_count,
                 data = train_rf)

summary(gam_model)

# RMSE on training set
gam_train_pred <- predict(gam_model, newdata = train_rf)
cat("GAM RMSE (train):", sqrt(mean((train_rf$fare_amount - gam_train_pred)^2)), "\n")

# Final submission
gam_pred <- predict(gam_model, newdata = test_dat)
outDat <- data.frame(uid = test_dat$uid, fare_amount = gam_pred)
write.csv(outDat, "W26P1_submission7.csv", row.names = FALSE)
