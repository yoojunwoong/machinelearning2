#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# In[3]:


favorite_load = pickle.load(open('./saves/favorite_save.pkl','rb'))
print(favorite_load)


# In[4]:


type(favorite_load)


# In[5]:


print(favorite_load['tiger'])


# In[11]:


autompg_lr = pickle.load(open('./saves/autompg_lr.pkl','rb'))
print(autompg_lr)


# In[12]:


print(type(autompg_lr))


# In[13]:
a = 3504.0
b = 8

import numpy as np
np.array([[a,b]])
pre = np.array([[a,b]])

print(autompg_lr.predict(pre))
print(autompg_lr.predict([[3504.0,8]]))


# In[ ]:




