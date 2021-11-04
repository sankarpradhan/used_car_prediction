#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os


# In[3]:


os.chdir ('C:\\Users\\sanka\\Desktop\\data science class\\data\\used_car_prediction')
os.getcwd()


# In[4]:


df=pd.read_csv('car data.csv')


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


print(df['Seller_Type'].unique())


# In[8]:


print(df['Transmission'].unique())


# In[9]:


print(df['Fuel_Type'].unique())


# In[10]:


print(df['Owner'].unique())


# In[11]:


#checking null value


# In[12]:


df.isnull().sum()


# In[13]:


df.describe()


# In[14]:


df.columns


# In[15]:


final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[16]:


final_dataset['Current_Year']=2021


# In[17]:


final_dataset


# In[18]:


final_dataset['no_of_years']=final_dataset['Current_Year']-final_dataset['Year']


# In[19]:


final_dataset.head()


# In[20]:


final_dataset.drop(['Current_Year'],axis=1,inplace=True)


# In[21]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[22]:


final_dataset.head()


# In[23]:


#onehotencoding using dummies()


# In[24]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True) #dummy variable trap


# In[25]:


final_dataset.head()


# In[26]:


final_dataset.corr()


# In[27]:


import seaborn as sns


# In[28]:


sns.pairplot(final_dataset)


# In[29]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

corrmat= final_dataset.corr()
top_corr_features= corrmat.index
plt.figure(figsize = (20,20))
g = sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[30]:


final_dataset.head()


# In[31]:


#independent and dependent featueres
x= final_dataset.iloc[:,1:]
y= final_dataset.iloc[:,0]


# In[32]:


x.head()


# In[33]:


y.head()


# In[34]:


#feature importance
from sklearn.ensemble import ExtraTreesRegressor
model= ExtraTreesRegressor()
model.fit(x,y)


# In[35]:


print(model.feature_importances_)


# In[36]:


#plt graph of featuresimportances for better visualization
feat_importances= pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[37]:


from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


# In[38]:


x_train.shape


# In[39]:


x_test.shape


# In[40]:


from sklearn.ensemble import RandomForestRegressor
rf_rando = RandomForestRegressor()


# In[41]:


## hypeparameters
import numpy as np
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=18 )]
print(n_estimators)


# In[ ]:





# In[42]:


#randomized search cv
#no. of trees in random forest
n_estimatorses =[int(x) for x in np.linspace(start=100, stop=1200, num = 12)]
#no. of features to consider at every split
max_features = ['auto','sqrt']
#maximum no. of levels in tree
max_depth = [int(x) for x in np.linspace(5,30,num =6)]
#max_depth.append(none)
#minumum no. of samples required to leaf node
min_samples_split = [2,5,10,15,100]
#minimum no. of samples required at each leaf node
min_samples_leaf = [1,2,5,10]


# In[43]:


from sklearn.model_selection import RandomizedSearchCV


# In[44]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[45]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()


# In[46]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[47]:


rf_random.fit(x_train,y_train)


# In[48]:


predictions = rf_random.predict(x_test)


# In[49]:


predictions


# In[50]:


sns.distplot(y_test-predictions)


# In[51]:


plt.scatter(y_test,predictions)


# In[52]:


from  sklearn.metrics import r2_score
r2_score(y_test,predictions)


# In[53]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# 

# In[54]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')
# dump information to that file
pickle.dump(rf_random, file)


# In[55]:


pip freeze > requirements.txt.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




