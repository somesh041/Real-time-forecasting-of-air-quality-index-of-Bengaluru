#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing the required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymannkendall as mk
import statsmodels.tsa.stattools as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import boxcox
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import math
from scipy.stats import variation
import sklearn
from sklearn.metrics import mean_squared_error
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams['figure.figsize']=(15,7)


# In[ ]:


# Importing the data and grouping it into monthly data 
data = pd.read_csv("data.csv", parse_dates = ['Date'], infer_datetime_format=True)
df_month =data.groupby(pd.Grouper(key='Date', freq='M')).mean()
df_month.reset_index('Date', inplace=True)
df_month


# # Data Visualization 

# In[ ]:


sns.lineplot(x ='Date', y= df['AQI_calculated'],data=df ,label='AQI',linewidth = 3.0,)
plt.legend(loc='best')
plt.title('AQI of Bengaluru from Jan-2017 to June-2022\n', fontdict={'fontsize': 16, 'fontweight' : 5, 'color' : 'red'})
plt.xticks(rotation = 90,fontweight="bold");


# In[ ]:


sns.boxplot(x=df_month['AQI_calculated'], data = df_month)


# In[ ]:


pd.plotting.lag_plot(df_month['AQI_calculated'], lag=1)
plt.show()


# In[ ]:


plot_acf(df_month['AQI_calculated']);


# In[ ]:


plot_pacf(df_month['AQI_calculated']);


# In[ ]:


# Considering the AQI column for the analysis 
df= df_month[['Date', 'AQI_calculated']]
df.set_index('Date', inplace=True)


# In[ ]:


# decomposiion of time series data 
add_decompose = seasonal_decompose(df, model="additive", period=12)  # additive decomposition
fig = add_decompose.plot()
plt.show()


# In[ ]:


mult_decompose = seasonal_decompose(df, model="multiplicative", period=12)  # multiplicative decomposition
fig = mult_decompose.plot()
plt.show()


# ## test for randomeness and test for trend

# In[ ]:


import scipy.integrate as SP
import math
def turningpoints(A):
    N = 0
    u = 0
    u_prev = 0
for i in range(1,66):
    u = A[i-1]-A[i] #Change between elements
if u < u_prev: #if change has gotten smaller
    N = N+1 #number of turning points increases
u_prev = u #set the change as the previous change
return N
if __name__ == "__main__":
    A = df['AQI_calculated']
print(turningpoints(A))


# In[ ]:


average = (2 * 66 - 4)/3
std_error = pow((16*66-29)/4, 0.5)
Z = (turningpoints(A)-average)/std_error
print(Z)
import scipy.stats as st
if abs(Z) >= st.norm.ppf(.05):
    print('We reject Ho and conclude that data is non random.')
else:
    print('We do not reject Ho and conclude that data is random.')


# In[ ]:


# spliiting the data into train and tes datasets for model building. 
train_len = 50
train = df[0 : train_len]
test = df[train_len : ]


# In[ ]:


#!pip install pymannkendall
import pymannkendall as mk
mk.original_test(df_month['AQI_calculated'])


# # Model fitting

# In[ ]:


## Holt Winters' additive method with trend and seasonality
y_hat_hwa = test.copy()
model = ExponentialSmoothing(np.asarray(train) ,seasonal_periods=12 ,trend='add', seasonal='add')
model_fit = model.fit(optimized=True)
print(model_fit.params)
y_hat_hwa['hw_forecast'] = model_fit.forecast(len(test))


# In[ ]:


plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(y_hat_hwa['hw_forecast'], label='Holt Winters\'s additive forecast')
plt.legend(loc='best')
plt.title('Holt Winters\' Additive Method')
plt.show()


# In[ ]:


rmse = np.sqrt(mean_squared_error(test, y_hat_hwa['hw_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['AQI_calculated']-y_hat_hwa['hw_forecast'])/test['AQI_calculated'])*100,2)

results = pd.DataFrame({'Method':['Holt Winters\'s additive forecast'], 'RMSE': [rmse],'MAPE': [mape] })
results


# In[ ]:


## Holt Winter's multiplicative method with trend and seasonality
y_hat_hwm = test.copy()
model = ExponentialSmoothing(np.asarray(train) ,seasonal_periods=12 ,trend='add', seasonal='mul')
model_fit = model.fit(optimized=True)
print(model_fit.params)
y_hat_hwm['hw_forecast'] = model_fit.forecast(len(test))


# In[ ]:


plt.plot( train, label='Train')
plt.plot(test, label='Test')
plt.plot(y_hat_hwm['hw_forecast'], label='Holt Winters\'s mulitplicative forecast')
plt.legend(loc='best')
plt.title('Holt Winters\' Mulitplicative Method')
plt.show()


# In[ ]:


rmse = np.sqrt(mean_squared_error(test, y_hat_hwm['hw_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['AQI_calculated']-y_hat_hwm['hw_forecast'])/test['AQI_calculated'])*100,2)

tempResults = pd.DataFrame({'Method':['Holt Winters\' multiplicative method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# In[ ]:


## ADF test for stationarity

adf_test = st.adfuller(df['AQI_calculated'])   
print('ADF Statistic: %f' % adf_test[0])
print('Critical Values @ 0.05: %.2f' % adf_test[4]['5%'])
print('p-value: %f' %adf_test[1])


# In[ ]:


## KPSS test for stationarity
from statsmodels.tsa.stattools import kpss
kpss_test = kpss(df['AQI_calculated'])

print('KPSS Statistic: %f' % kpss_test[0])
print('Critical Values @ 0.05: %.2f' % kpss_test[3]['5%'])
print('p-value: %f' % kpss_test[1])


# In[ ]:


## Box Cox transformation to make variance constant
df_boxcox = pd.Series(boxcox(df['AQI_calculated'], lmbda=0), index = df.index)
plt.figure(figsize=(12,4))
plt.plot(df_boxcox, label='After Box Cox tranformation')
plt.legend(loc='best')
plt.title('After Box Cox transform')
plt.show()


# In[ ]:


## Differencing to remove trend
df_boxcox_diff = pd.Series(df_boxcox - df_boxcox.shift(), df.index)

plt.plot(df_boxcox_diff, label='After Box Cox tranformation and differencing')
plt.legend(loc='best')
plt.title('After Box Cox transform and differencing')
plt.show()


# In[ ]:


df_boxcox_diff.dropna(inplace=True)


# In[ ]:


df_boxcox_diff.isna().sum()


# In[ ]:


adf_test = st.adfuller(df_boxcox_diff)

print('ADF Statistic: %f' % adf_test[0])
print('Critical Values @ 0.05: %.2f' % adf_test[4]['5%'])
print('p-value: %f' % adf_test[1])


# In[ ]:


kpss_test = kpss(df_boxcox_diff)

print('KPSS Statistic: %f' % kpss_test[0])
print('Critical Values @ 0.05: %.2f' % kpss_test[3]['5%'])
print('p-value: %f' % kpss_test[1])


# In[ ]:


train_df_boxcox = df_boxcox[:train_len]
test_df_boxcox = df_boxcox[train_len:]
train_df_boxcox_diff = df_boxcox_diff[:train_len-1]
test_df_boxcox_diff = df_boxcox_diff[train_len-1:]


# In[ ]:


print(train_df_boxcox.shape)
print(test_df_boxcox.shape)
print(train_df_boxcox_diff.shape)
print(test_df_boxcox_diff.shape)


# In[ ]:


## Auto regressive integrated moving average (ARIMA)
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train_df_boxcox_diff, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.params)


# In[ ]:


# Recover original time series forecast
y_hat_arima = df_boxcox_diff.copy()
y_hat_arima['arima_forecast_boxcox_diff'] = model_fit.predict(df_boxcox_diff.index.min(), df_boxcox_diff.index.max())
y_hat_arima['arima_forecast_boxcox'] = y_hat_arima['arima_forecast_boxcox_diff'].cumsum()
y_hat_arima['arima_forecast_boxcox'] = y_hat_arima['arima_forecast_boxcox'].add(df_boxcox[0])
y_hat_arima['arima_forecast'] = np.exp(y_hat_arima['arima_forecast_boxcox'])


# In[ ]:


plt.plot(train['AQI_calculated'], label='Train')
plt.plot(test['AQI_calculated'], label='Test')
plt.plot(y_hat_arima['arima_forecast'][test.index.min():], label='ARIMA forecast')
plt.legend(loc='best')
plt.title('Autoregressive integrated moving average (ARIMA) method')
plt.show()


# In[ ]:


# Calculate RMSE and MAPE

rmse = np.sqrt(mean_squared_error(test, y_hat_arima['arima_forecast'][test.index.min():])).round(2)
mape = np.round(np.mean(np.abs(test['AQI_calculated']-y_hat_arima['arima_forecast'][test.index.min():])/test['AQI_calculated'])*100,2)

tempResults = pd.DataFrame({'Method':['Autoregressive integrated moving average (ARIMA) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# In[ ]:


## Seasonal auto regressive integrated moving average (SARIMA)
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train_df_boxcox, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) 
model_fit = model.fit()
print(model_fit.params)


# In[ ]:


# Recover original time series forecast
y_hat_sarima = df_boxcox_diff.copy()
y_hat_sarima['sarima_forecast_boxcox'] = model_fit.predict(df_boxcox_diff.index.min(), df_boxcox_diff.index.max())
y_hat_sarima['sarima_forecast'] = np.exp(y_hat_sarima['sarima_forecast_boxcox'])


# In[ ]:


plt.plot(train['AQI_calculated'], label='Train')
plt.plot(test['AQI_calculated'], label='Test')
plt.plot(y_hat_sarima['sarima_forecast'][test.index.min():], label='SARIMA forecast')
plt.legend(loc='best')
plt.title('SARIMA(1,1,1,12) method')
plt.show()


# In[ ]:



rmse = np.sqrt(mean_squared_error(test['AQI_calculated'], y_hat_sarima['sarima_forecast'][test.index.min():])).round(2)
mape = np.round(np.mean(np.abs(test['AQI_calculated']-y_hat_sarima['sarima_forecast'][test.index.min():])/test['AQI_calculated'])*100,2)

tempResults = pd.DataFrame({'Method':['SARIMA(1,1,1,12) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# In[ ]:


from pmdarima import auto_arima
stepwise_fit = auto_arima(df['AQI_calculated'], start_p = 0, start_q = 0,
                          max_p = 3, max_q = 3, m = 12,
                          start_P = 0, seasonal = True,
                          d = None, D = None, trace = True,
                          error_action ='ignore',   # we don't want to know if an order does not work
                          suppress_warnings = True,  # we don't want convergence warnings
                          stepwise = True)           # set to stepwise
  
# To print the summary
stepwise_fit.summary()


# In[ ]:


model_1 = SARIMAX(train_df_boxcox, order=(0,1,1), seasonal_order=(1, 0, 1, 12)) 
model_1_fit = model_1.fit()
print(model_1_fit.params)


# In[ ]:


y_hat_sarima = df_boxcox_diff.copy()
y_hat_sarima['SARIMA(0,1,1)(1,0,1)_forecast_boxcox'] = model_1_fit.predict(df_boxcox_diff.index.min(), df_boxcox_diff.index.max())
y_hat_sarima['SARIMA(0,1,1)(1,0,1)_forecast'] = np.exp(y_hat_sarima['SARIMA(0,1,1)(1,0,1)_forecast_boxcox'])


# In[ ]:


plt.plot(train['AQI_calculated'], label='Train')
plt.plot(test['AQI_calculated'], label='Test')
plt.plot(y_hat_sarima['SARIMA(0,1,1)(1,0,1)_forecast'][test.index.min():], label='SARIMA(0,1,1)(1,0,1) forecast')
plt.legend(loc='best')
plt.title('SARIMA(0,1,1)(1,0,1) method')
plt.show()


# In[ ]:



rmse = np.sqrt(mean_squared_error(test['AQI_calculated'], y_hat_sarima['SARIMA(0,1,1)(1,0,1)_forecast'][test.index.min():])).round(2)
mape = np.round(np.mean(np.abs(test['AQI_calculated']-y_hat_sarima['SARIMA(0,1,1)(1,0,1)_forecast'][test.index.min():])/test['AQI_calculated'])*100,2)

tempResults = pd.DataFrame({'Method':['SARIMA(0,1,1)(1,0,1) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# In[ ]:


model_1_fit.plot_diagnostics();


# In[ ]:


print(model_1_fit.summary().tables[1])


# In[ ]:


print(model_1_fit.aicc)
print(model_1_fit.bic)


# In[ ]:


step_fit = auto_arima(df['AQI_calculated'], start_p = 0, start_q = 0,
                          max_p = 3, max_q = 3, m = 12,
                          start_P = 0, seasonal = True,
                          d = None, D = 1, trace = True,
                          error_action ='ignore',   # we don't want to know if an order does not work
                          suppress_warnings = True,  # we don't want convergence warnings
                          stepwise = True) 

step_fit.summary()


# In[ ]:


model_2 = SARIMAX(train_df_boxcox, order=(0,0,0), seasonal_order=(0, 1, 1, 12)) 
model_2_fit = model_2.fit()
print(model_2_fit.params)


# In[ ]:


y_hat_sarima = df_boxcox_diff.copy()
y_hat_sarima['SARIMA(0,0,0)(0,1,1)_forecast_boxcox'] = model_2_fit.predict(df_boxcox_diff.index.min(), df_boxcox_diff.index.max())
y_hat_sarima['SARIMA(0,0,0)(0,1,1)_forecast'] = np.exp(y_hat_sarima['SARIMA(0,0,0)(0,1,1)_forecast_boxcox'])


# In[ ]:


plt.plot(train['AQI_calculated'], label='Train')
plt.plot(test['AQI_calculated'], label='Test')
plt.plot(y_hat_sarima['SARIMA(0,0,0)(0,1,1)_forecast'][test.index.min():], label='SARIMA(0,0,0)(0,1,1) forecast')
plt.legend(loc='best')
plt.title('SARIMA(0,0,0)(0,1,1) method')
plt.show()


# In[ ]:


rmse = np.sqrt(mean_squared_error(test['AQI_calculated'], y_hat_sarima['SARIMA(0,0,0)(0,1,1)_forecast'][test.index.min():])).round(2)
mape = np.round(np.mean(np.abs(test['AQI_calculated']-y_hat_sarima['SARIMA(0,0,0)(0,1,1)_forecast'][test.index.min():])/test['AQI_calculated'])*100,2)

tempResults = pd.DataFrame({'Method':['SARIMA(0,0,0)(0,1,1) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# In[ ]:


print(model_2_fit.aicc)
print(model_2_fit.bic)


# In[ ]:


print(model_2_fit.summary().tables[1])


# In[ ]:


model_2_fit.plot_diagnostics();


# ## Forecating the out of sample values 

# ### Forecasting using statsmaodels.tsa.SARIMAX()

# In[ ]:


mod = SARIMAX(df, order =(0,0,0), seasonal_order =(0,1,1,12), trend = 'ct' )
res = mod.fit()
print(res.summary())


# In[ ]:


print(res.forecast(steps=10))


# In[ ]:


plt.plot(df, label='Actual AQI')
plt.plot(fcast['mean'], label='Forecasted AQI')
plt.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='green', alpha=0.05);
plt.legend(loc='best')
plt.title('Forecast plot of AQI')
plt.show()


# ### Forecasting using pmdarima

# In[ ]:


preds, conf_int = step_fit.predict(n_periods= 10,
                                return_conf_int=True)


# In[ ]:


print(preds)


# In[ ]:


plt.plot(df, label='Actual AQI')
plt.plot(preds, label='Forecasted AQI')
plt.fill_between(preds.index, conf_int[:, 0], conf_int[:, 1],
                     alpha=0.05, color='b');
plt.legend(loc='best')
plt.title('Forecast plot of AQI')
plt.show()


# ## Cross validation of fitted model and Forecasted values

# ### Cross validation of fitted model and Forecasted values using SARIMAX

# In[ ]:


# fit model parameters w/ training sample
training_obs = int(len(df) * 0.8)
training_df = df[:training_obs]
# Setup forecasts
nforecasts = 10
forecasts = {}

# Get the number of initial training observations
nobs = len(df)
n_init_training = int(nobs * 0.8)

# Create model for initial training sample, fit parameters
init_training_df = df.iloc[:n_init_training]
mod = SARIMAX(training_df, order=(0,0,0),seasonal_order=(0,1,1,12), trend='ct')
res = mod.fit()

# Save initial forecast
forecasts[training_df.index[-1]] = res.forecast(steps=nforecasts)

# Step through the rest of the sample
for t in range(n_init_training, nobs):
    # Update the results by appending the next observation
    updated_df = df.iloc[t:t+1]
    res = res.extend(updated_df)

    # Save the new set of forecasts
    forecasts[updated_df.index[0]] = res.forecast(steps=nforecasts)

# Combine all forecasts into a dataframe
forecasts = pd.concat(forecasts, axis=1)

print(forecasts.iloc[:5, :5])


# In[ ]:


forecast_errors = forecasts.apply(lambda column: df['AQI_calculated'] - column).reindex(forecasts.index)

print(forecast_errors.iloc[:5, :5])


# In[ ]:


def flatten(column):
    return column.dropna().reset_index(drop=True)

flattened = forecast_errors.apply(flatten)
flattened.index = (flattened.index + 1).rename('horizon')

print(flattened.iloc[:3, :5])


# In[ ]:


# Compute the root mean square error
rmse = (flattened**2).mean(axis=1)**0.5

print(rmse)


# ### Cross validation of fitted model and Forecasted values using pmdarima

# In[ ]:


data = df
from pmdarima import model_selection
train, test = model_selection.train_test_split(data, train_size=50)

# Even though we have a dedicated train/test split, we can (and should) still
# use cross-validation on our training set to get a good estimate of the model
# performance. We can choose which model is better based on how it performs
# over various folds.
model1 = pm.ARIMA(order=(0,1,1), seasonal_order = (1,0,1,12),trend = 'c',suppress_warnings=True)
model2 = pm.ARIMA(order=(0,0,0),seasonal_order=(0, 1, 1, 12),trend = 'c',suppress_warnings=True)
cv = pm.model_selection.RollingForecastCV(step=4, h=2)

model1_cv_scores = model_selection.cross_val_score(
    model1, train, scoring='mean_squared_error', cv=cv, verbose=2)

model2_cv_scores = model_selection.cross_val_score(
    model2, train, scoring='mean_squared_error', cv=cv, verbose=2)

print("Model 1 CV scores: {}".format(model1_cv_scores.tolist()))
print("Model 2 CV scores: {}".format(model2_cv_scores.tolist()))

# Pick based on which has a lower mean error rate
m1_average_error = np.average(model1_cv_scores)
m2_average_error = np.average(model2_cv_scores)
errors = [m1_average_error, m2_average_error]
models = [model1, model2]

# print out the answer
better_index = np.argmin(errors)  # type: int
print("Lowest average mean_squared_error: {} (model{})".format(
    errors[better_index], better_index + 1))
print("Best model: {}".format(models[better_index]))

