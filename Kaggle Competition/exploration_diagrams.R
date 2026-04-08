library(glmnet)
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

train_dat[train_dat$distance <=0, ]

# ── Prepare model matrices ─────────────────────────────────────────────────────

features <- c("pickup_longitude", "pickup_latitude",
              "dropoff_longitude", "dropoff_latitude",
              "distance", "hour", "dow", "passenger_count")

par(mfrow = c(3, 3))

# Distribution of target
hist(train_dat$fare_amount, main = "fare_amount", 
     col = "lightblue", breaks = 50, xlab = "")

# Diagram used in introduction
# Histograms for each feature
for (f in features) {
  hist(train_dat[[f]], main = f, 
       col = "lightblue", breaks = 50, xlab = "")
}

par(mfrow = c(3, 3))

# Average fare plot for questions 2, 3 and 4 of exploratory data analysis
for (f in features) {
  # round feature to reduce number of unique values
  rounded <- round(train_dat[[f]], 1)
  avg_fare <- tapply(train_dat$fare_amount, rounded, mean)
  
  plot(as.numeric(names(avg_fare)), avg_fare,
       main = paste("Avg fare vs", f),
       xlab = f, ylab = "Avg fare ($)",
       type = "l", col = "blue", lwd = 2)
}

# For Question 1 of exploratory data analysis
# Most common pickup latitude band
lat_freq <- sort(table(round(train_dat$pickup_latitude, 3)), decreasing = TRUE)
head(lat_freq,1)

# Most common pickup longitude band
lon_freq <- sort(table(round(train_dat$pickup_longitude, 3)), decreasing = TRUE)
head(lon_freq,1)

