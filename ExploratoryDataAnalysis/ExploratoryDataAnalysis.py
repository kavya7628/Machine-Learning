#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_iris


# In[3]:


iris_images= load_iris()


# In[4]:


iris_images.data


# In[5]:


iris_images.target


# In[6]:


X = iris_images.data
y = iris_images.target


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[10]:


model.fit(X_train,y_train)


# In[11]:


model.predict(X_test)


# In[12]:


accuracy_lg=model.score(X_test, y_test)

print("Accuracy of Logistic Regression:",accuracy_lg)


# In[13]:


from sklearn.tree import DecisionTreeClassifier


# In[14]:


model1 = DecisionTreeClassifier()


# In[15]:


model1.fit(X_train,y_train)


# In[16]:


model1.predict(X_test)


# In[17]:


accuracy_dt = model1.score(X_test, y_test)

print("Accuracy of Decision Tree:",accuracy_dt)


# In[18]:


from sklearn import neural_network


# In[31]:


model2 = neural_network.MLPClassifier(hidden_layer_sizes=(4,),activation = 'relu',solver ='adam',learning_rate='constant',learning_rate_init=0.02,max_iter=1000, shuffle=True, random_state=None)


# In[32]:


model2.fit(X_train,y_train)


# In[33]:


model2.predict(X_test)


# In[34]:


accuracy = model2.score(X_test, y_test)
print("Accuracy of Neural Net:",accuracy)


# In[ ]:




