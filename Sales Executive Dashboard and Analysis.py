#!/usr/bin/env python
# coding: utf-8

# Take Home Challenge: Sales Executive Dashboard and Analysis
# 
# The objective of this case study is to help give you a better sense of the types of problems that we face on a daily basis. As the preliminary Analytics team within ABC, we are frequently asked to help solve business challenges that generally involve measuring performance and impact to our customers. This specific case study is focused on an analysis for a fictional DVD distribution company. We estimate this case study should take 2-3 days to complete.
# Good luck!
# 
# Disclaimer: The datasets and scenario provided for this case study are completely fictional. 
# 

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
pd.options.display.float_format = '{:.2f}'.format


# In[2]:


#importing data
data = pd.read_csv("C:/Users/bushi/OneDrive/Bureau/Case Studies/Analytics_Case_Study - Sales_Data.csv", 
                   skipinitialspace = True)

#inspecting the first 5 rows of the dataframe
data.head()


# In[3]:


data.shape

# Features

iD:	Unique ID For a transaction
first_name:	 First Name of customer
last_name:	 Last Name of customer
Email:	 email of customer
Gender:	 Gender of customer
Unit_quantity:	 # of DVD’s purchased
Cost of goods:	 Cost of DVD to produce at the time of purchase
Sale price per unit:	 US Dollar cost of each DVD Sold
Order date:	 Order date of Transaction
Delivery_date:	 Actual Date of delivery
Promise_date:	 Delivery Date promised to customer
Country:	 Country of sale transaction
State:	 State of sale transaction
DVD_Title:	 Title of DVD sold
Movie_Genre:	 Detailed genre of DVD Sold
Discount_Applied:	 Binary variable that signifies whether a discount was applied. Discount is 5% of the listed sale_price for each unit during the transaction


# In[4]:


# Summary of dataset
data.info()


# In[5]:


# Remove certain columns
data_2 = data.copy()
data_2.drop(['id', 'first_name', 'last_name', 'email', 'Country'], 1, inplace = True)

#removing special character in amount columns
data_2['cost_of_goods'] = data_2['cost_of_goods'].str.replace('$', '')
data_2['sale_price_per_unit'] = data_2['sale_price_per_unit'].str.replace('$', '')

data_2.head()


# In[6]:


#converting amount column into float
data_2["cost_of_goods"] = data_2["cost_of_goods"].astype(float)
data_2["sale_price_per_unit"] = data_2["sale_price_per_unit"].astype(float)

data_2.info()


# In[7]:


#converting date column into datetime format
data_2["order_date"] = pd.to_datetime(data["order_date"])
data_2["delivery_date"] = pd.to_datetime(data["delivery_date"])
data_2["promise_date"] = pd.to_datetime(data["promise_date"])

data_2.dtypes


# In[8]:


# Computing revenue, profit, total cost and ROI
# Profit = (sale_price_per_unit - cost_of_goods) * unit_quantity
# ROI = sale_price_per_unit / cost_of_goods
data_2['Revenue'] = data_2.sale_price_per_unit * data_2.unit_quantity
data_2['Total_Cost'] = data_2.cost_of_goods * data_2.unit_quantity
data_2['Profit'] = data_2.Revenue - data_2.Total_Cost
data_2['ROI'] = data_2.Profit / data_2.Total_Cost

data_2.head()


# In[9]:


# Extracting years and months and period
data_2['Year'] = data_2['delivery_date'].dt.year
data_2['Month'] = data_2['delivery_date'].dt.month
data_2['Yr_Month'] = data_2['delivery_date'].dt.strftime('%Y-%m')

data_2.head()


# In[10]:


# expecting datatype
data_2.dtypes


# In[11]:


data_2.hist(figsize=(20,12), bins=200)
plt.show()


# In[12]:


data_2.describe()


# In[13]:


data_2.describe(include = 'object')


# In[14]:


#first and last name add no value, ID will be used in lieu of both
#Dataset is related to the US so can remove country field
df = data_2.copy()

df.head()


# In[15]:


# Removing missing value in Movie_Genre and Discount column; which represent only 12% of the sample dataset
# May explore another method i.e. replacing missing data 
df_nonmissing = df.dropna(axis=0)
df_nonmissing.info()


# In[16]:


df_nonmissing.describe()


# In[17]:


df_nonmissing.describe(include='object')


# In[18]:


df_nonmissing.corr()


# In[19]:


df_trend = df_nonmissing.groupby("Yr_Month").sum()[["unit_quantity", "cost_of_goods", "sale_price_per_unit", 'Revenue', 'Total_Cost', "Profit"]].reset_index()
df_trend


# In[20]:


df_trend.info()


# In[21]:


df_trend.describe()


# In[22]:


plt.figure(figsize=(15, 6))
plt.plot(df_trend['Yr_Month'], df_trend['Profit'], label = 'Profit')
plt.plot(df_trend['Yr_Month'], df_trend['Revenue'], label = 'Revenue')
plt.xticks(rotation = 'vertical', size = 10)
plt.legend()
plt.show()


# ### Conclusion 1 - Overall, sales and profits are going down overtime

# In[ ]:





# ## Driver of Revenue

# In[23]:


df_trend_2 = df_nonmissing.groupby(["gender", "Yr_Month", "State", "Discount_Applied", "Movie_Genre"]).sum()[["unit_quantity", "sale_price_per_unit","Revenue"]].reset_index()
df_trend_2


# In[24]:


df_trend_2 = df_nonmissing.groupby("State").sum()[["unit_quantity", "Revenue"]].reset_index()

# Top 10 states by cumulative profit 
df_trend_2.sort_values("Revenue", ascending = False)


# In[25]:


plt.figure(figsize=(15, 6))
plt.plot(df_trend_2["State"], df_trend_2['Revenue'])
plt.xticks(rotation = 'vertical', size = 10)
plt.show()


# In[26]:


df_trend_2 = df_nonmissing.groupby("gender").sum()[["unit_quantity", "Revenue"].reset_index()
df_trend_2


# In[ ]:


df_trend_2 = df_nonmissing.groupby("Discount_Applied").sum()[['sale_price_per_unit', 'Revenue']].reset_index()
df_trend_2


# In[ ]:


plt.figure(figsize=(15, 6))
plt.plot(df_trend_2['Discount_Applied'], df_trend_2['Revenue'])
plt.xticks(rotation = 'vertical', size = 10)
plt.show()


# In[ ]:



df_revdvr = df_nonmissing[['State', 'Movie_Genre', 'Discount_Applied', 'sale_price_per_unit', 'Revenue']]
df_revdvr


# In[ ]:


# Transformation of object variables
obj_df = df_revdvr.select_dtypes(include=['object']).copy()


# In[ ]:


from sklearn import preprocessing
lb_make = preprocessing.LabelEncoder()
for col in obj_df.columns.values:
    df_revdvr[f'{col}_new'] = lb_make.fit_transform(df_revdvr[f'{col}'])


# In[ ]:


# Inspect new dataframe
df_revdvr.head()


# In[ ]:


# Deleting object column
df_revdvr.drop(['State', 'Movie_Genre', 'Discount_Applied'], 1, inplace = True)


# In[ ]:


df_revdvr.info()


# In[ ]:


# Building a predictive model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


x = df_revdvr.iloc[:, 0:2]
y = df_revdvr['Revenue']
x_train, x_test, y_train, y_test = train_test_split(x, y)


# In[ ]:


# Create and fit the model for prediction
lin = LinearRegression()
lin.fit(x_train, y_train)
y_pred = lin.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


coef = lin.coef_
component = pd.DataFrame(zip(x.columns, coef), columns = ['component', 'value'])
component = component.append({'component':'intercept', 'value':lin.intercept_}, ignore_index=True)
component


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Top 20 Profitable 
df_nonmissing = df_nonmissing.sort_values("Profit", ascending = False)
df_nonmissing.head(20)


# In[ ]:


plt.figure(figsize=(15, 6))
plt.plot(df_trend_2['gender'], df_trend_2['Profit'])
plt.xticks(rotation = 'vertical', size = 10)
plt.show()


# In[ ]:


agg = df_nonmissing.groupby("gender")["Profit"].mean()
agg


# In[ ]:


agg = df_nonmissing.groupby(["gender", "Discount_Applied"])["Profit"].mean()
agg


# In[ ]:


agg = df_nonmissing.groupby(["gender", "Discount_Applied", "Movie_Genre"])["Profit"].mean()
print(agg)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


agg = df_nonmissing.groupby(["gender", "State"])["Profit"].mean()
agg


# In[ ]:


agg = df_nonmissing.groupby(["gender", "Year"])["Profit"].mean()
agg


# In[ ]:





# In[ ]:


agg = df_nonmissing.pivot_table(values = "Profit", index = "gender", columns = ["Year"], margins = True)
agg


# In[ ]:


agg = df_nonmissing.pivot_table(values = "Profit", index = "State", columns = ["Year"], margins = True)
agg


# In[ ]:





# In[ ]:


agg = df_nonmissing.groupby(["gender"])["sale_price_per_unit"].mean()
agg


# In[ ]:


agg = df_nonmissing.groupby(["gender", "Year"])["sale_price_per_unit"].mean()
agg


# In[ ]:


agg = df_nonmissing.groupby(["gender", "Year"])["sale_price_per_unit"].agg([min, max])
agg


# In[ ]:





# In[ ]:





# ## Plotting the Data

# In[ ]:


import seaborn as sns


# In[ ]:


# Creating new dataframe
df_1 = df_nonmissing[['delivery_date', 'Profit', 'ROI', 'Year', 'Month']]
df_1.info()

# Display
df_1.head()


# In[ ]:


df_1.Profit.plot(figsize=(20,2), title = "Profit")

plt.show()


# In[ ]:


df_1.corr()


# In[ ]:



df_1.Profit.plot(figsize=(20,5), title = "Profit")

plt.show()


# In[ ]:


# Setting delivery date as index
#df_1 = df_1.set_index("delivery_date", inplace = True)

#df_1.info()
# Display
#df_1.head()


# In[ ]:


# Line chart 
sns.lineplot(df_1.delivery_date, df_1.Profit)
plt.xticks(months)
#plt.show()


# In[ ]:


# Line chart 
sns.lineplot(x = 'delivery_date', y = 'Profit', data = df_1)
plt.show()


# In[ ]:





# In[ ]:


# Line chart showing daily global streams of each song 
sns.lineplot(data=df_1)


# In[ ]:





# In[ ]:


df_1.Profit.plot(figsize=(20,2), title = "Profit")
#df_nonmissing_1.Profit.plot(figsize=(20,5), title = "Profit")
plt.show()


# In[ ]:


df_nonmissing.head()


# In[ ]:


# movie with highest profit
df_nonmissing[df_nonmissing.Profit == df_nonmissing.Profit.max()]


# In[ ]:


# movie with lowest profit
df_nonmissing[df_nonmissing.Profit == df_nonmissing.Profit.min()]


# In[ ]:


# movie with highest ROI
df_nonmissing[df_nonmissing.ROI == df_nonmissing.ROI.max()]


# In[ ]:


# movie with lowest ROI
df_nonmissing[df_nonmissing.ROI == df_nonmissing.ROI.min()]


# In[ ]:




