#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd


# In[12]:


import numpy as np


# In[13]:


from sklearn import linear_model 


# In[14]:


df = pd.read_csv("homeprices.csv")


# In[15]:


df.info()


# In[16]:


df


# In[17]:


model = linear_model.LinearRegression()


# In[18]:


model.fit(df[["area"]].values,df[["price"]].values)


# In[19]:


model.coef_


# In[20]:


model.intercept_


# In[25]:


model.predict(np.array([[3300]]))


# In[26]:


import pickle


# In[27]:


with open("model_pickle","wb") as file:
    pickle.dump(model,file)


# In[28]:


with open("model_pickle","rb") as file:
    pickledModel = pickle.load(file)


# In[29]:


pickledModel


# In[30]:


pickledModel.predict(np.array([[3300]]))


# In[32]:


from sklearn.externals import joblib


# In[33]:


joblib.dump(model,"model_pickle_using_joblib")


# In[34]:


modelPickledUsingJoblib = joblib.load("model_pickle_using_joblib")


# In[35]:


modelPickledUsingJoblib.predict(np.array([[3300]]))


# In[ ]:




