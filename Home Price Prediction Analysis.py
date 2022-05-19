#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error ,mean_absolute_error


# In[2]:


cd F:\Dataset\Done projects\Home Price Prediction Analysis


# # Part-1: data Exploration and Pre-processing

# In[3]:


# Load the dataset


# In[4]:


df = pd.read_csv("Python_Linear_Regres.csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.shape


# In[ ]:


# Print all the column names


# In[8]:


df.columns


# In[ ]:


# Describe the data


# In[9]:


df.describe()


# # remove unwonted columns like address, date, postcode, YearBuilt, lattitude, longtitude

# In[10]:


df= df.drop(['Address','Date','Postcode','YearBuilt','Lattitude','Longtitude'],axis = 1)
df


# # Find the count of null value in each column

# In[11]:


df.isnull().sum()


# # Fill the null value of property count, distance, Bedroom2, Bathroom, Car replace with 0

# In[12]:


df = df.fillna({
    'Propertycount' : 0,
    'Distance' : 0,
    'Bedroom2' : 0,
    'Bathroom' : 0,
    'Car' : 0
})


# In[13]:


df


# In[14]:


df.isnull().sum()


# # Fill Null value of land size and bidding area columns with Mean

# In[15]:


df[['Landsize','BuildingArea']]


# In[16]:


df['Landsize'] = df['Landsize'].replace(np.nan, df['Landsize'].mean())


# In[ ]:


# or 


# In[ ]:


df['Landsize'] = df['Landsize'].fillna('?')
data = df['Landsize'].loc[df['Landsize'] != '?']
mean = data.astype(int).mean()
df['Landsize'] = df['Landsize'].replace('?',mean).astype(int)
df['Landsize']


# In[17]:


df['BuildingArea'] = df['BuildingArea'].replace(np.nan, df['BuildingArea'].mean())


# In[ ]:


# or


# In[ ]:


df['BuildingArea'] = df['BuildingArea'].fillna('?')
data = df['BuildingArea'].loc[df['BuildingArea'] != '?']
mean = data.astype(int).mean()
df['BuildingArea'] = df['BuildingArea'].replace('?',mean).astype(int)
df['BuildingArea']


# In[18]:


df.isnull().sum()


# In[19]:


df.dropna(inplace=True)


# In[20]:


df.shape


# # Find the unique value in method column

# In[21]:


df.Method.unique()


# In[22]:


df['SellerG'].unique()

# SellerG = df[["SellerG"]]

# SellerG = pd.get_dummies(SellerG, drop_first= True)

# SellerG.head()


# In[23]:


df['CouncilArea'].unique()
# CouncilArea = df[["CouncilArea"]]

# CouncilArea = pd.get_dummies(CouncilArea, drop_first= True)

# CouncilArea.head()


# In[24]:


df['Regionname'].unique()
# Regionname = df[["Regionname"]]

# Regionname = pd.get_dummies(Regionname, drop_first= True)

# Regionname.head()


# In[25]:


df.columns


# # Create a dummy data for categorical data

# In[26]:


df = pd.get_dummies(df, drop_first=True)


# In[27]:


df.info()


# In[28]:


df.shape


# In[29]:


df.head()


# In[30]:


df.columns


# # Part-2: Working with Model

# In[ ]:


#  Create the target data and feature data where target data is price


# In[31]:


x = df.drop(['Price'],axis = 1).astype(int)
x.head()


# In[32]:


y = df.Price.astype(int)
y.head()


# In[33]:


x.shape, y.shape


# # Spliting training and testing dataset

# In[34]:


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.2,random_state=41)


# In[35]:


X_train.shape, Y_train.shape  , X_test.shape, Y_test.shape


# In[36]:


X_test


# In[37]:


X_train


# In[ ]:





# # Create a linear regression model for Target and feature data

# In[38]:


from sklearn.linear_model import LinearRegression


# In[39]:


# train the model
model = LinearRegression()
model.fit(X_train,Y_train)


# In[40]:


price_predicted = model.predict(X_test)


# In[41]:


price_predicted = np.array(price_predicted)
price_predicted


# In[42]:


Y_test


# # Check if the model is overfitting or underfitting or it is accurate

# In[43]:


model.score(X_test,Y_test)


# In[44]:


model.score(X_train,Y_train)


# # The model is overfitting then apply ridge and lasso regression algorithms

# In[52]:


from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha = 49)
ridge_reg.fit(X_train,Y_train)


# In[54]:


ridge_reg.score(X_test,Y_test)


# In[55]:


ridge_reg.score(X_train,Y_train)


# In[56]:


from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha = 49)
lasso_reg.fit(X_train,Y_train)


# In[57]:


lasso_reg.score(X_test,Y_test)


# In[58]:


lasso_reg.score(X_train,Y_train)


# # Display Mean Squared Error

# In[59]:


mse = mean_squared_error(Y_test,price_predicted)
mse


# # Display Mean Absolute Error

# In[60]:


mae = mean_absolute_error(Y_test,price_predicted)
mae


# # Display Root mean Squared error

# In[61]:


rmse = np.sqrt(mean_squared_error(Y_test,price_predicted))
rmse


# # Display R2 score

# In[62]:


r2 = r2_score(Y_test,price_predicted)
r2


# In[ ]:




