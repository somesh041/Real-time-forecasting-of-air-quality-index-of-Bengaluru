#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
plt.rcParams['figure.figsize']=(15,7)


# In[2]:


df = pd.read_csv('Bengaluru_aqi.csv', parse_dates = ['Date'])    # importing the dataset


# In[3]:


df   # view the dataset


# In[4]:


df.info()


# In[5]:


df.drop(df.tail(1).index, inplace = True)


# In[6]:


df['SO2'] = pd.to_numeric(df['SO2'], errors='coerce')  # convert object type to float type.


# In[7]:


df.info()


# In[8]:


df.describe() #summary statistics of the dataset


# # Data cleaning

# In[9]:


df.isnull().sum()     # checking the missing values


# In[10]:


df['PM2.5'] = df['PM2.5'].interpolate(option='spline')


# In[11]:


df['PM10'] = df['PM10'].interpolate(option='spline')


# In[12]:


df['NO'] = df['NO'].interpolate(pad='slinear')


# In[13]:


df['NO2'] = df['NO2'].interpolate(pad='slinear')


# In[14]:


df['NH3'] = df['NH3'].interpolate(option='spline')


# In[15]:


df['SO2'] = df['SO2'].interpolate(pad='slinear')


# In[16]:


df['O3'] = df['O3'].interpolate(option='spline')


# In[17]:


df['toluene'] = df['toluene'].interpolate(option='spline')


# In[18]:


df['benzene'] = df['benzene'].interpolate(option='spline')


# In[19]:


df['temp'] = df['temp'].interpolate(metod = 'linear')


# In[20]:


df['SR'] = df['SR'].interpolate(option='spline')


# In[21]:


df.isnull().sum()


# In[22]:


df['NH3'] = df['NH3'].interpolate(pad='slinear', limit_direction='backward')


# In[23]:


df['temp'] = df['temp'].fillna(df['temp'].mean())


# In[24]:


df.isnull().sum()


# # Calculating Sub indexes

# In[25]:


# PM2.5 Sub-Index calculation
def get_PM25_subindex(x):
    if x <= 30:
        return x * 50 / 30
    elif x > 30 and x <= 60:
        return 51 + (x - 31) * 49 / 29
    elif x > 60 and x <= 90:
        return 101 + (x - 61) * 99 / 29
    elif x> 91 and x <= 120:
        return 201 + (x - 91) * 99 / 29
    elif x > 121 and x <= 250:
        return 301 + (x - 121) * 99 / 129
    elif x > 250:
        return 401 + (x - 251) * 99 / 129
    else:
        return 0

df["PM2.5_SubIndex"] = df["PM2.5"].apply(lambda x: get_PM25_subindex(x))


# In[26]:


#  PM10 Sub-Index calculation
def get_PM10_subindex(x):
    if x <= 50:
        return x
    elif x> 51 and x <= 100:
        return x
    elif x> 101 and x <= 250:
        return 101 + (x - 101) * 99 / 149
    elif x> 251 and x <= 350:
        return 201 + (x - 251) 
    elif x> 351 and x <= 430:
        return 301 + (x - 351) * 99 / 79
    elif x> 431:
        return 401 + (x - 431) * 99 / 79
    else:
        return 0

df["PM10_SubIndex"] = df["PM10"].apply(lambda x: get_PM10_subindex(x))


# In[27]:


def get_NOX_subindex(x):
    if x <= 40:
        return x * 5/4
    elif x> 41 and x <= 80:
        return 51 + (x - 41) * 49 / 39
    elif x> 81 and x <= 180:
        return 101 + (x - 81) 
    elif x> 181 and x <= 280:
        return 201 + (x - 181)  
    elif x> 281 and x <= 400:
        return 301 + (x - 281) * 99 / 119
    elif x> 401:
        return 401 + (x - 401) * 99 / 119
    else:
        return 0

df["NO_SubIndex"] = df["NO"].apply(lambda x: get_NOX_subindex(x))


# In[28]:


df["NO2_SubIndex"] = df["NO2"].apply(lambda x: get_NOX_subindex(x))


# In[29]:


def get_SO2_subindex(x):
    if x <= 40:
        return x * 5/4
    elif x> 41 and x <= 80:
        return 51 + (x - 41) * 49 / 39
    elif x> 81 and x <= 380:
        return 101 + (x - 81) * 99 / 299 
    elif x> 381 and x <= 800:
        return 201 + (x - 381) * 99 / 419  
    elif x> 801 and x <= 1600:
        return 301 + (x - 801) * 99 / 799
    elif x> 1601:
        return 401 + (x - 1601) * 99 / 1099
    else:
        return 0

df["SO2_SubIndex"] = df["SO2"].apply(lambda x: get_SO2_subindex(x))


# In[30]:


def get_NH3_subindex(x):
    if x <= 200:
        return x * 50 / 200
    elif x> 201 and x <= 400:
        return 51 + (x - 201) * 49 / 199
    elif x> 401 and x <= 800:
        return 101 + (x - 401) * 99 / 399
    elif x> 801 and  x <= 1200:
        return 201 + (x - 801) * 99 / 399
    elif x> 1201 and x <= 1800:
        return 301 + (x - 1201) * 99 / 599
    elif x> 1800:
        return 401 + (x - 1801) * 99 / 599
    else:
        return 0

df["NH3_SubIndex"] = df["NH3"].apply(lambda x: get_NH3_subindex(x))


# In[31]:


def get_CO_subindex(x):
    x = x/3
    if x <= 1:
        return x * 50 / 1
    elif x> 1.1 and x <= 2:
        return 51 + (x - 1.1) * 50 / 0.9
    elif x> 2.1 and x <= 10:
        return 1001 + (x - 2.1) * 99 / 7.9
    elif x> 10.1 and x <= 17:
        return 201 + (x - 10.1) * 99 / 6.9
    elif x> 18 and x <= 34:
        return 301 + (x - 18) * 99 / 16.9
    elif x> 34:
        return 401 + (x - 35) * 99 / 16.9
    else:
        return 0

df["CO_SubIndex"] = df["CO"].apply(lambda x: get_CO_subindex(x))


# In[32]:


## O3 Sub-Index calculation
def get_O3_subindex(x):
    x = x / 8
    if x <= 50:
        return x * 50 / 50
    elif x> 51 and x <= 100:
        return 51 + (x - 51)
    elif x> 101 and x <= 168:
        return 101 + (x - 101) * 99 / 67
    elif x> 169 and x <= 208:
        return 201 + (x - 169) * 99 / 39
    elif x> 209 and x <= 748:
        return 301 + (x - 209) * 99 / 539
    elif x> 748:
        return 400 + (x - 400) * 100 / 539
    else:
        return 0

df["O3_SubIndex"] = df["O3"].apply(lambda x: get_O3_subindex(x))


# # Calculating AQI

# In[33]:


df["AQI_calculated"] = round(df[["PM2.5_SubIndex", "PM10_SubIndex","NO_SubIndex","NO2_SubIndex", "SO2_SubIndex",
                                 "NH3_SubIndex", "CO_SubIndex", "O3_SubIndex"]].max(axis = 1))


# In[34]:


df


# In[35]:


df.describe()


# In[36]:


def AQI_Category(x):
    if x <= 50:
        return "Good"
    elif x <= 100:
        return "Satisfactory"
    elif x <= 200:
        return "Moderate"
    elif x <= 300:
        return "Poor"
    elif x <= 400:
        return "severe"
    elif x > 400:
        return "Hazardous"
    else:
        return np.NaN


# In[37]:


df["AQI_Category"] = df["AQI_calculated"].apply(lambda x: AQI_Category(x))


# In[38]:


df['AQI_Category'].value_counts()

