#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("Groceries_dataset[1].csv")


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.notnull().sum()


# In[7]:


df.isna().sum()


# In[8]:


df.head()


# In[9]:


#setting index as Date
df.set_index('Date',inplace = True)


# In[10]:


df.head()


# In[ ]:


#converting date into a particular format
df.index=pd.to_datetime(df.index)


# In[12]:


df.head()


# In[13]:


df.shape


# In[14]:


#gathering information about products
total_item = len(df)
total_days = len(np.unique(df.index.date))
total_months = len(np.unique(df.index.year))
print(total_item,total_days,total_months)


# # Total 38765 items  sold in 728 days throughout 24 months

# In[15]:


plt.figure(figsize=(15,5))
sns.barplot(x = df.itemDescription.value_counts().head(20).index, y = df.itemDescription.value_counts().head(20).values, palette = 'gnuplot')
plt.xlabel('itemDescription', size = 15)
plt.xticks(rotation=45)
plt.ylabel('Count of Items', size = 15)
plt.title('Top 20 Items purchased by customers', color = 'green', size = 20)
plt.show()


# In[16]:


df['itemDescription'].value_counts()


# In[17]:


#grouping dataset to form a list of products bought by same customer on same date
df=df.groupby(['Member_number','Date'])['itemDescription'].apply(lambda x: list(x))


# In[18]:


df.head(10)


# In[19]:


#apriori takes list as an input, hence converting dtaset to a list
transactions = df.values.tolist()
transactions[:10]


# In[25]:


#applying apriori
from apyori import apriori
rules = apriori(transactions, min_support=0.00030,min_confidence = 0.05,min_lift = 2,min_length = 2)
results = list(rules)
results


# In[26]:


def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
ordered_results = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])


# In[27]:


ordered_results


# In[ ]:




