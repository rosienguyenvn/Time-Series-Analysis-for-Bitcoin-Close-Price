
## USE FORECAST AND ZOO LIBRARY.

# Install.packages("forecast")
library(forecast)
library(zoo)

## CREATE DATA FRAME FOR Bitcoin STOCK PRICES. 

# Set working directory for locating files.
#setwd("/Users/kavya/Desktop/CSUEB/673/Project/Final_files")

# Set working directory for locating files.
setwd("/Users/anjullunkad/CSUEB/Semester 3/TimeSeries/Project-BitcoinForecasting")


# Create data frame.
bitc.data <- read.csv("BTC_USD.csv")

# See the first and last 6 records of the file for bitcoin data.
head(bitc.data)
tail(bitc.data)

## USE ts() FUNCTION TO CREATE TIME SERIES DATASET FOR bitcoin CLOSE PRICES.

# Create time series data for daily close stock prices, consider frequency 
# equal to 1. 
close.price.ts <- ts(bitc.data$Close,
                     start = 1, freq = 1)
close.price.ts
length(close.price.ts)

## Use plot() to plot time series data  
# Use plot() function to create plot For Close Price.

rdate <- as.Date(bitc.data$Date, "%m/%d/%y")
plot(close.price.ts~rdate, type = "l",    xlab = "Time in mm-yy format", 
                    ylab = "Bitcoin Close Price, in $", ylim = c (0, 80000), 
                    main = "Bitcoin Close Price in USD from July 2017 to July 2021",
                    xaxt = "n", bty = "l", lwd = 2, col="blue")
axis (1, rdate, format(rdate,"%m-%y"))




## TEST PREDICTABILITY OF BITCOIN CLOSE STOCK PRICES.

# Use Arima() function to fit AR(1) model for Bitcoin close prices.
# The ARIMA model of order = c(1,0,0) gives an AR(1) model.
close.price.ar1<- Arima(close.price.ts, order = c(1,0,0))
summary(close.price.ar1)

# Since AR(1) coefficient is very close to 1, 
# checking if time series data has random walk through Hypothesis test (z-statistic approach)
# B1 = 0.9983
# Null Hypothesis Ho: B1 = 1, Alternate Hypothesis Ha: B1 <>1
# z-statistic = (B1 -Hypothesized value of B1)/ s.e = (0.9983-1)/0.0015
z.stat <- (0.9983-1)/0.0015
z.stat  # z- statistic value is -1.13333

# Calculate p-value for z-statistic
p.value <- pnorm(z.stat)
p.value  # p-value is 0.12854 cannot reject Null Hypothesis at 95% confidence.
# It means that B1 = 1, data has random walk issues indicative of data may not be predictable.

#Using first differencing method

# Create first difference of ClosePrice data using diff() function.
diff.close.price <- diff(close.price.ts, lag = 1)
diff.close.price


# Use Acf() function to identify autocorreltion for first differenced
# Close Price and plot autocorrelation for different lags 
# (up to maximum of 12 lags).
Acf(diff.close.price, lag.max = 12, 
    main = "Autocorrelation for Bitcoin Differenced Close Prices")


############################################################################################################

# Use Acf() function to identify autocorrelation and plot autocorrelation
# for different lags (up to maximum of 12).
Acf(close.price.ts, lag.max = 12, main = "Autocorrelation for Bitcoin Close price")

############################################################################################################

## CREATE TIME SERIES PARTITION.

# Define the numbers of days in the training and validation sets,
# nTrain and nValid, respectively.
# Total number of period length(close.price.ts) = 1458
# nvalid = 219
# nTrain = 1239
nValid <- 219
length(close.price.ts)
nTrain <- length(close.price.ts) - nValid
nTrain
train.ts <- window(close.price.ts, start = c(1,1), end = c(1, nTrain))
valid.ts <- window(close.price.ts, start = c(1, nTrain + 1), 
                   end = c(1, nTrain + nValid))


############################################################################################################

## FIT AUTO ARIMA MODEL.

# Use auto.arima() function to fit ARIMA model.
# Use summary() to show auto ARIMA model and its parameters.
train.auto.arima <- auto.arima(train.ts)
summary(train.auto.arima)

# Apply forecast() function to make predictions for ts with 
# auto ARIMA model in validation set.  
train.auto.arima.pred <- forecast(train.auto.arima, h = nValid, level = 0)
train.auto.arima.pred

# Using Acf() function, create autocorrelation chart of auto ARIMA 
# model residuals.
Acf(train.auto.arima$residuals, lag.max = 12, 
    main = "Autocorrelations of Auto ARIMA Model Residuals")

round(accuracy(train.auto.arima.pred, valid.ts), 3)



## FIT AUTO ARIMA MODELS FOR ENTIRE DATA SET. 

# Use auto.arima() function to fit ARIMA model for entire data set.
# use summary() to show auto ARIMA model and its parameters for entire data set.
auto.arima <- auto.arima(close.price.ts)
summary(auto.arima)

# Apply forecast() function to make predictions for ts with 
# auto ARIMA model for the future 14 periods. 
auto.arima.pred <- forecast(auto.arima, h = 14, level = 0)
auto.arima.pred

# Use Acf() function to create autocorrelation chart of auto ARIMA 
# model residuals.
Acf(auto.arima$residuals, lag.max = 12, 
    main = "Autocorrelations of Auto ARIMA Model Residuals")

#Use accuracy function 
round(accuracy(auto.arima.pred$fitted, close.price.ts), 3)

############################################################################################################

# Create Holt's exponential smoothing for partitioned data.
# Use ets() function with model = "ZZZ", i.e., automated selection 
# error, trend, and seasonality options.
# Use optimal alpha, beta, & gamma to fit Holt's over the training period.
h.ZZZ <- ets(train.ts, model = "ZZZ")
h.ZZZ 

# Use forecast() function to make predictions using this Holt's model with 
# validation period (nValid). 
# Show predictions in tabular format.
h.ZZZ.pred <- forecast(h.ZZZ, h = nValid, level = 0)
h.ZZZ.pred

#Use accuracy function 
round(accuracy(h.ZZZ.pred$mean, valid.ts), 3)


## FORECAST WITH HOLT'S MODEL USING ENTIRE DATA SET INTO
## THE FUTURE FOR 14 PERIODS.

# Create Holt's  exponential smoothing for Bitcoin closing price 
# Use ets() function with model = "ZZZ", to identify the best Holt's option
# and optimal alpha, beta, & gamma to fit HW for the entire data period.
H.ZZZ <- ets(close.price.ts, model = "ZZZ")
H.ZZZ 

# Use forecast() function to make predictions using this Holt's model for
# 14 days into the future.
H.ZZZ.pred <- forecast(H.ZZZ, h = 14 , level = 0)
H.ZZZ.pred


# Identify performance measures for Holt's forecast.
round(accuracy(H.ZZZ.pred$fitted, close.price.ts), 3)

############################################################################################################

# Two- level model : for partitions
# Regression model with linear trend and AR(1) for residuals
# Linear trend
train.lin <- tslm(train.ts ~ trend)
summary(train.lin)


#Develop an AR(1) model for the regression residuals
res.lin.ar1 <- Arima(train.lin$residuals, order = c(1,0,0))
summary(res.lin.ar1)

# Use Acf() function to identify autocorrealtion for the training 
# residual of residuals and plot autocorrelation for different lags 
# (up to maximum of 12)
Acf(res.lin.ar1$residuals, lag.max = 8,   main = "Autocorrelation AR(1) for Bitcoin Close Price Training Residuals of Residuals")


# Use forecast() function to make prediction for validation set using egression model with linear trend.
train.lin.pred <- forecast(train.lin, h = nValid, level = 0)
train.lin.pred


# Use forecast() function to make prediction of residuals in validation set.
res.ar1.pred <- forecast(res.lin.ar1, h = nValid, level = 0)
res.ar1.pred


#Create a two-level forecasting model (regression model with linear trend + AR(1) model for residuals) for the validation period 
valid.two.level.lin.pred <- train.lin.pred$mean + res.ar1.pred$mean
valid.lin.df <- data.frame(valid.ts, train.lin.pred$mean, 
                       res.ar1.pred$mean, valid.two.level.lin.pred )
names(valid.lin.df) <- c("Validation", "Reg.Forecast", 
                     "AR(1)Forecast", "Combined.Forecast")
valid.lin.df


# Identify performance measures 
# # Two- level model : for partitions
# Regression model with linear trend and AR(1) for residuals
round(accuracy(valid.two.level.lin.pred, valid.ts), 3)


#Two-level forecasting for entire data set

## FIT REGRESSION MODEL WITH LINEAR TREND and AR(1) for residuals
## FOR ENTIRE DATASET. FORECAST AND PLOT DATA, AND MEASURE ACCURACY.

# Use tslm() function to create linear trend 
lin.trend <- tslm(close.price.ts ~ trend)

# See summary of linear trend equation and associated parameters.
summary(lin.trend)

# Apply forecast() function to make predictions with linear trend
# model into the future 14 periods.  
lin.trend.pred <- forecast(lin.trend, h = 14, level = 0)
lin.trend.pred


# Use Arima() function to fit AR(1) model for regression residuals.
# The ARIMA model order of order = c(1,0,0) gives an AR(1) model.
# Use forecast() function to make prediction of residuals into the future 14 periods.
lin.residual.ar1 <- Arima(lin.trend$residuals, order = c(1,0,0))
lin.residual.ar1.pred <- forecast(lin.residual.ar1, h = 14, level = 0)


# Use summary() to identify parameters of AR(1) model.
summary(lin.residual.ar1)

# Use Acf() function to identify autocorrealtion for the residual of residuals 
# and plot autocorrelation for different lags (up to maximum of 12).
Acf(lin.residual.ar1$residuals, lag.max = 12, 
    main = "Autocorrelation for Bitcoin Residuals of Residuals for Entire Data Set")


# Identify forecast for the future 14 periods as sum of linear trend  model
# and AR(1) model for residuals.
lin.trend.ar1.pred <- lin.trend.pred$mean + lin.residual.ar1.pred$mean
lin.trend.ar1.pred


# Create a data table with linear trend  forecast for 14 future periods,
# AR(1) model for residuals for 14 future periods, and combined two-level forecast for
# 14 future periods. 
table.lin.df <- data.frame(lin.trend.pred$mean, 
                           lin.residual.ar1.pred$mean, lin.trend.ar1.pred)
names(table.lin.df) <- c("Reg.Forecast", "AR(1)Forecast","Combined.Forecast")
table.lin.df

############################################################################################################

# Two- level model : for partitions
# Regression model with Quadtratic trend and AR(1) for residuals

# Regression mode with quadratic trend 
train.quad <- tslm(train.ts ~ trend + I(trend^2))

# See summary of quadratic trend model and associated parameters.
summary(train.quad)


#Develop an AR(1) model for the regression residuals
res.ar1 <- Arima(train.quad$residuals, order = c(1,0,0))
summary(res.ar1)

Acf(res.ar1$residuals, lag.max = 8,   main = "Autocorrelation AR(1) for Bitcoin Close Price Training Residuals of Residuals")


#AR(1) for regression residuals equation : et = -54.7636 + 0.6590 * et-1
# Use forecast() function to make prediction of residuals in validation set.
res.ar1.pred <- forecast(res.ar1, h = nValid, level = 0)
res.ar1.pred

train.quad.pred <- forecast(train.quad, h = nValid, level = 0)
train.quad.pred



#Create a two-level forecasting model (regression model with quadratic trend + AR(1) model for residuals) for the validation period 
valid.two.level.quad.pred <- train.quad.pred$mean + res.ar1.pred$mean


valid.quad.df <- data.frame(valid.ts, train.quad.pred$mean, 
                       res.ar1.pred$mean, valid.two.level.quad.pred)
names(valid.quad.df) <- c("Validation", "Reg.Forecast", 
                     "AR(1)Forecast", "Combined.Forecast")
valid.quad.df


# Identify performance measures 
# # Two- level model : for partitions
# Regression model with quadratic trend and AR(1) for residuals
round(accuracy(valid.two.level.quad.pred, valid.ts), 3)


#Two-level forecasting for entire data set

## FIT REGRESSION MODEL WITH QUADRATIC TREND
## FOR ENTIRE DATASET. FORECAST AND PLOT DATA, AND MEASURE ACCURACY.

# Use tslm() function to create quadratic trend 
quad.trend <- tslm(close.price.ts ~ trend + I(trend^2) )

# See summary of quadratic trend equation and associated parameters.
summary(quad.trend)

# Apply forecast() function to make predictions with quadratic trend  
# model into the future 14 periods.  
quad.trend.pred <- forecast(quad.trend, h = 14, level = 0)
quad.trend.pred


# Use Arima() function to fit AR(1) model for regression residuals.
# The ARIMA model order of order = c(1,0,0) gives an AR(1) model.
# Use forecast() function to make prediction of residuals into the future 14 periods.
quad.residual.ar1 <- Arima(quad.trend$residuals, order = c(1,0,0))
quad.residual.ar1.pred <- forecast(quad.residual.ar1, h = 14, level = 0)


# Use summary() to identify parameters of AR(1) model.
summary(quad.residual.ar1)

# Use Acf() function to identify autocorrealtion for the residual of residuals 
# and plot autocorrelation for different lags (up to maximum of 12).
Acf(quad.residual.ar1$residuals, lag.max = 12, 
    main = "Autocorrelation for Bitcoin Residuals of Residuals for Entire Data Set")


# Identify forecast for the future 14 periods as sum of quadratic trend model
# and AR(1) model for residuals.
quad.trend.ar1.pred <- quad.trend.pred$mean + quad.residual.ar1.pred$mean
quad.trend.ar1.pred


# Create a data table with quadratic trend and seasonal forecast for 14 future periods,
# AR(1) model for residuals for 14 future periods, and combined two-level forecast for
# 14 future periods. 
table.quad.df <- data.frame(quad.trend.pred$mean, 
                       quad.residual.ar1.pred$mean, quad.trend.ar1.pred)
names(table.quad.df) <- c("Reg.Forecast", "AR(1)Forecast","Combined.Forecast")
table.quad.df


############################################################################################################

# Two-level forecasting using Auto ARIMA in level 1 and trailing MA for residuals in Level 2


## FIT AUTO ARIMA MODELS FOR validation partition 

# Use auto.arima() function to fit ARIMA model.
# Use summary() to show auto ARIMA model and its parameters.
train.auto.arima <- auto.arima(train.ts)
summary(train.auto.arima)

# Apply forecast() function to make predictions using auto ARIMA model in validation set.  
train.auto.arima.pred <- forecast(train.auto.arima, h = nValid, level = 0)
train.auto.arima.pred

# Use Acf() function to create autocorrelation chart of auto ARIMA 
# model residuals.
Acf(auto.arima$residuals, lag.max = 12, 
    main = "Autocorrelations of Auto ARIMA Model Residuals")


# Identify and display auto ARIMA residuals
train.auto.arima.res <- train.auto.arima$residuals
train.auto.arima.res

# Use trailing MA to forecast residuals
train.arima.ma.trail.res <- rollmean(train.auto.arima.res, k = 2, align = "right")
train.arima.ma.trail.res

# Create forecast for trailing MA residuals for validation period.
train.arima.ma.trail.res.pred <- forecast(train.arima.ma.trail.res, h = nValid, level = 0)
train.arima.ma.trail.res.pred


# Develop 2-level forecast for training partition by combining 
# auto ARIMA forecast and trailing MA for residuals to forecast for validation period
train.arima.ma.fst.2level <- train.auto.arima.pred$mean + train.arima.ma.trail.res.pred$mean
train.arima.ma.fst.2level

# Create a table with validation partition data, auto ARIMA forecast, trailing MA for residuals,
# and total forecast for validation partition
train.future14.df <- data.frame(valid.ts, train.auto.arima.pred$mean, train.arima.ma.trail.res.pred$mean, 
                          train.arima.ma.fst.2level)
names(train.future14.df) <- c("Validation", "AutoArima.Fst", "MA.Residuals.Fst", "Combined.Fst")
train.future14.df

round(accuracy(train.auto.arima.pred$mean + train.arima.ma.trail.res.pred$mean, valid.ts), 3)


# Two-level forecasting using Auto ARIMA in level 1 and trailing MA for residuals in Level 2 for entire data set
## FIT AUTO ARIMA MODELS FOR ENTIRE DATA SET. 

# Use auto.arima() function to fit ARIMA model for entire data set.
# use summary() to show auto ARIMA model and its parameters for entire data set.
auto.arima <- auto.arima(close.price.ts)
summary(auto.arima)

# Apply forecast() function to make predictions for ts with 
# auto ARIMA model for the future 14 periods. 
auto.arima.pred <- forecast(auto.arima, h = 14, level = 0)
auto.arima.pred

# Use Acf() function to create autocorrelation chart of auto ARIMA 
# model residuals.
Acf(auto.arima$residuals, lag.max = 12, 
    main = "Autocorrelations of Auto ARIMA Model Residuals")

# Identify and display regression residuals for entire data set.
auto.arima.res <- auto.arima$residuals
auto.arima.res

# Use trailing MA to forecast residuals for entire data set.
arima.ma.trail.res <- rollmean(auto.arima.res, k = 2, align = "right")
arima.ma.trail.res

# Create forecast for trailing MA residuals for future 14 periods.
arima.ma.trail.res.pred <- forecast(arima.ma.trail.res, h = 14, level = 0)
arima.ma.trail.res.pred


# Develop 2-level forecast for future 14 periods by combining 
# regression forecast and trailing MA for residuals for future
# 14 periods.
arima.ma.fst.2level <- auto.arima.pred$mean + arima.ma.trail.res.pred$mean
arima.ma.fst.2level

# Create a table with regression forecast, trailing MA for residuals,
# and total forecast for future 14 periods.
future14.df <- data.frame(auto.arima.pred$mean, arima.ma.trail.res.pred$mean, 
                          arima.ma.fst.2level)
names(future14.df) <- c("AutoArima.Fst", "MA.Residuals.Fst", "Combined.Fst")
future14.df



############################################################################################################

# Use accuracy() function to identify common accuracy measures for:
# Accuracy measures for all the models - for validation partition
# (1) Auto ARIMA
# (2) Holt's model
# (3) two-level model (linear trend  + AR(1) model for residuals),
# (4) two-level model (quadratic trend  + AR(1) model for residuals)
# (5) Two-level model ( Auto ARIMA + Trailing MA for residuals)
# (6) Naive forecast.

round(accuracy(train.auto.arima.pred$mean, valid.ts), 3)

round(accuracy(h.ZZZ.pred$mean, valid.ts), 3)

round(accuracy(train.lin.pred$mean + res.ar1.pred$mean, valid.ts), 3)

round(accuracy(train.quad.pred$mean + res.ar1.pred$mean, valid.ts), 3)

round(accuracy(train.auto.arima.pred$mean + train.arima.ma.trail.res.pred$mean, valid.ts), 3)

round(accuracy((naive(train.ts))$mean, valid.ts), 3)

# Use accuracy() function to identify common accuracy measures for:
# Accuracy measures for all the models - for entire data set
# (1) Auto ARIMA
# (2) Holt's model
# (3) two-level model (linear trend  + AR(1) model for residuals),
# (4) two-level model (quadratic trend  + AR(1) model for residuals)
# (5) Two-level model ( Auto ARIMA + Trailing MA for residuals)
# (6) Naive forecast.

round(accuracy(auto.arima.pred$fitted, close.price.ts), 3)

round(accuracy(H.ZZZ.pred$fitted, close.price.ts), 3)

round(accuracy(lin.trend$fitted + lin.residual.ar1$fitted, close.price.ts), 3)

round(accuracy(quad.trend$fitted + quad.residual.ar1$fitted, close.price.ts), 3)

round(accuracy(auto.arima.pred$fitted + arima.ma.trail.res, close.price.ts), 3)

round(accuracy((naive(close.price.ts))$fitted, close.price.ts), 3)

