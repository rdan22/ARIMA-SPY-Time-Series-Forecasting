'''
Autoregressive Integrated Moving Average (ARIMA) Model converts non-stationary data to stationary data before working on it. 
It is one of the most popular models to predict linear time series data.
The data shows the stock price of SPY from 2015-03-20 till 2020-03-19. 
The goal is to train an ARIMA model with optimal parameters that will forecast the closing price of the stocks on the test data.
'''
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np

#load dataset
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('path to SPY.csv',sep=',', index_col='Date', parse_dates=['Date'], date_parser=dateparse).fillna(0)

#visualize the per-day closing price of the stock
#plot close price
plt.figure(figsize=(10,6))''
plt.grid(True)d
plt.xlabel('Dates')
plt.ylabel('Close Prices')
plt.plot(data['Close'])
plt.title('SPY Inc. closing price')
plt.show()

#plot the scatterplot
df_close = data['Close']
df_close.plot(style='k.')
plt.title('Scatter plot of closing price')
plt.show()

'''
Also, a given time series is thought to consist of three systematic components including level, trend, seasonality, and one non-systematic component called noise.
These components are defined as follows:
	Level: The average value in the series.
	Trend: The increasing or decreasing value in the series.
	Seasonality: The repeating short-term cycle in the series.
	Noise: The random variation in the series.
First, we need to check if a series is stationary or not because time series analysis only works with stationary data.
'''
'''
ADF (Augmented Dickey-Fuller) Test:
The Dickey-Fuller test is one of the most popular statistical tests. It can be used to determine the presence of unit root in the series, 
and hence help us understand if the series is stationary or not. The null and alternate hypothesis of this test is:
	Null Hypothesis: The series has a unit root (value of a =1)
	Alternate Hypothesis: The series has no unit root.
If we fail to reject the null hypothesis, we can say that the series is non-stationary. This means that the series can be linear or difference stationary.
If both mean and standard deviation are flat lines(constant mean and constant variance), the series becomes stationary.
'''
#Test for stationarity
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    #output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
    
test_stationarity(df_close)

#Through the above graph, we can see the increasing mean and standard deviation and hence our series is not stationary.
#We see that the p-value is greater than 0.05 so we cannot reject the Null hypothesis. Also, the test statistics is greater than the critical values, so the data is non-stationary.
'''
Results of dickey fuller test
Test Statistics                  -1.648747
p-value                           0.457694
No. of lags used                  9.000000
Number of observations used    1249.000000
critical value (1%)              -3.435596
critical value (5%)              -2.863857
critical value (10%)             -2.568004
'''
#In order to perform a time series analysis, we may need to separate seasonality and trend from our series. The resultant series will become stationary through this process.
result = seasonal_decompose(df_close, model='multiplicative', freq = 30)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(16, 9)

'''
we start by taking a log of the series to reduce the magnitude of the values and reduce the rising trend in the series. 
Then after getting the log of the series, we find the rolling average of the series. 
A rolling average is calculated by taking input for the past 12 months and giving a mean consumption value at every point further ahead in series.
'''
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
df_log = np.log(df_close)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average')
plt.plot(std_dev, color ="black", label = "Standard Deviation")
plt.plot(moving_avg, color="red", label = "Mean")
plt.legend()
plt.show()

'''
Now we are going to create an ARIMA model and will train it with the closing price of the stock on the train data. 
So let us split the data into training and test set and visualize it.
'''
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_log, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()

model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find             optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
print(model_autoARIMA.summary())

'''
Performing stepwise search to minimize aic
Fit ARIMA: (0, 1, 0)x(0, 0, 0, 0) (constant=True); AIC=-7514.720, BIC=-7504.661, Time=0.096 seconds
Fit ARIMA: (1, 1, 0)x(0, 0, 0, 0) (constant=True); AIC=-7513.076, BIC=-7497.988, Time=0.204 seconds
Fit ARIMA: (0, 1, 1)x(0, 0, 0, 0) (constant=True); AIC=-7513.117, BIC=-7498.029, Time=0.442 seconds
Fit ARIMA: (0, 1, 0)x(0, 0, 0, 0) (constant=False); AIC=-7515.144, BIC=-7510.115, Time=0.051 seconds
Fit ARIMA: (1, 1, 1)x(0, 0, 0, 0) (constant=True); AIC=-7511.128, BIC=-7491.012, Time=0.296 seconds
Total fit time: 1.098 seconds
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                 1130
Model:               SARIMAX(0, 1, 0)   Log Likelihood                3758.572
Date:                Thu, 19 Mar 2020   AIC                          -7515.144
Time:                        20:35:03   BIC                          -7510.115
Sample:                             0   HQIC                         -7513.244
                               - 1130                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
sigma2      7.508e-05   1.83e-06     40.990      0.000    7.15e-05    7.87e-05
===================================================================================
Ljung-Box (Q):                       50.27   Jarque-Bera (JB):               824.95
Prob(Q):                              0.13   Prob(JB):                         0.00
Heteroskedasticity (H):               1.01   Skew:                            -0.56
Prob(H) (two-sided):                  0.91   Kurtosis:                         7.04
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
'''
#Before moving forward, let's review the residual plots from auto ARIMA.
model_autoARIMA.plot_diagnostics(figsize=(15,8))
plt.show()

'''
We interpret as follows:
Top left: The residual errors seem to fluctuate around a mean of zero and have a uniform variance.
Top Right: The density plot suggest normal distribution with mean zero.
Bottom left: All the dots should fall perfectly in line with the red line. Any significant deviations would imply the distribution is skewed.
Bottom Right: The Correlogram, aka, ACF plot shows the residual errors are not autocorrelated. 
Any autocorrelation would imply that there is some pattern in the residual errors which are not explained in the model. So you will need to look for more X's (predictors) to the model.

Overall, it seems to be a good fit. let's start forecasting the stock prices.
'''
model = ARIMA(train_data, order=(3, 1, 2))  
fitted = model.fit(disp=-1)  
print(fitted.summary())
'''
							ARIMA Model Results                              
==============================================================================
Dep. Variable:                D.Close   No. Observations:                 1129
Model:                 ARIMA(3, 1, 2)   Log Likelihood                3766.464
Method:                       css-mle   S.D. of innovations              0.009
Date:                Thu, 19 Mar 2020   AIC                          -7518.928
Time:                        20:38:24   BIC                          -7483.724
Sample:                             1   HQIC                         -7505.627
                                                                              
=================================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
---------------------------------------------------------------------------------
const             0.0004   4.62e-05      8.076      0.000       0.000       0.000
ar.L1.D.Close     0.1180      0.091      1.292      0.196      -0.061       0.297
ar.L2.D.Close     0.8055      0.096      8.367      0.000       0.617       0.994
ar.L3.D.Close     0.0497      0.031      1.586      0.113      -0.012       0.111
ma.L1.D.Close    -0.1369      0.087     -1.568      0.117      -0.308       0.034
ma.L2.D.Close    -0.8631      0.087     -9.886      0.000      -1.034      -0.692
                                    Roots                                    
=============================================================================
                  Real          Imaginary           Modulus         Frequency
-----------------------------------------------------------------------------
AR.1            1.0142           +0.0000j            1.0142            0.0000
AR.2           -1.2414           +0.0000j            1.2414            0.5000
AR.3          -15.9889           +0.0000j           15.9889            0.5000
MA.1            1.0000           +0.0000j            1.0000            0.0000
MA.2           -1.1586           +0.0000j            1.1586            0.5000
-----------------------------------------------------------------------------
'''
#Now let's start forecast the stock prices on the test dataset keeping 95% confidence level.
# Forecast
fc, se, conf = fitted.forecast(252, alpha=0.05)  # 95% confidence
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_data, label='training')
plt.plot(test_data, color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('SPY Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()

#Not too shabby. Let us also check the commonly used accuracy metrics to judge forecast results:
mse = mean_squared_error(test_data, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))

'''
MSE: 0.004071023488844017
MAE: 0.04440193749959564
RMSE: 0.06380457263271981
MAPE: 0.007781946062028573

Around 0.77% MAPE (Mean Absolute Percentage Error) implies the model is about 99.23% accurate in predicting the test set observations.

'''

'''
OVERALL RESULTS

Results of dickey fuller test
Test Statistics                  -1.648747
p-value                           0.457694
No. of lags used                  9.000000
Number of observations used    1249.000000
critical value (1%)              -3.435596
critical value (5%)              -2.863857
critical value (10%)             -2.568004
dtype: float64
No handles with labels found to put in legend.
Performing stepwise search to minimize aic
Fit ARIMA: (0, 1, 0)x(0, 0, 0, 0) (constant=True); AIC=-7514.720, BIC=-7504.661, Time=0.089 seconds
Fit ARIMA: (1, 1, 0)x(0, 0, 0, 0) (constant=True); AIC=-7513.076, BIC=-7497.988, Time=0.199 seconds
Fit ARIMA: (0, 1, 1)x(0, 0, 0, 0) (constant=True); AIC=-7513.117, BIC=-7498.029, Time=0.442 seconds
Fit ARIMA: (0, 1, 0)x(0, 0, 0, 0) (constant=False); AIC=-7515.144, BIC=-7510.115, Time=0.051 seconds
Fit ARIMA: (1, 1, 1)x(0, 0, 0, 0) (constant=True); AIC=-7511.128, BIC=-7491.012, Time=0.299 seconds
Total fit time: 1.087 seconds
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                 1130
Model:               SARIMAX(0, 1, 0)   Log Likelihood                3758.572
Date:                Thu, 19 Mar 2020   AIC                          -7515.144
Time:                        20:41:42   BIC                          -7510.115
Sample:                             0   HQIC                         -7513.244
                               - 1130                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
sigma2      7.508e-05   1.83e-06     40.990      0.000    7.15e-05    7.87e-05
===================================================================================
Ljung-Box (Q):                       50.27   Jarque-Bera (JB):               824.95
Prob(Q):                              0.13   Prob(JB):                         0.00
Heteroskedasticity (H):               1.01   Skew:                            -0.56
Prob(H) (two-sided):                  0.91   Kurtosis:                         7.04
===================================================================================

                             ARIMA Model Results                              
==============================================================================
Dep. Variable:                D.Close   No. Observations:                 1129
Model:                 ARIMA(3, 1, 2)   Log Likelihood                3766.464
Method:                       css-mle   S.D. of innovations              0.009
Date:                Thu, 19 Mar 2020   AIC                          -7518.928
Time:                        20:41:46   BIC                          -7483.724
Sample:                             1   HQIC                         -7505.627
                                                                              
=================================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
---------------------------------------------------------------------------------
const             0.0004   4.62e-05      8.076      0.000       0.000       0.000
ar.L1.D.Close     0.1180      0.091      1.292      0.196      -0.061       0.297
ar.L2.D.Close     0.8055      0.096      8.367      0.000       0.617       0.994
ar.L3.D.Close     0.0497      0.031      1.586      0.113      -0.012       0.111
ma.L1.D.Close    -0.1369      0.087     -1.568      0.117      -0.308       0.034
ma.L2.D.Close    -0.8631      0.087     -9.886      0.000      -1.034      -0.692
                                    Roots                                    
=============================================================================
                  Real          Imaginary           Modulus         Frequency
-----------------------------------------------------------------------------
AR.1            1.0142           +0.0000j            1.0142            0.0000
AR.2           -1.2414           +0.0000j            1.2414            0.5000
AR.3          -15.9889           +0.0000j           15.9889            0.5000
MA.1            1.0000           +0.0000j            1.0000            0.0000
MA.2           -1.1586           +0.0000j            1.1586            0.5000
-----------------------------------------------------------------------------
MSE: 0.004071023488844017
MAE: 0.04440193749959564
RMSE: 0.06380457263271981
MAPE: 0.007781946062028573

'''