#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning Model Building Pipeline: Machine Learning Model Build
# 
# In the following videos, we will take you through a practical example of each one of the steps in the Machine Learning model building pipeline, which we described in the previous lectures. There will be a notebook for each one of the Machine Learning Pipeline steps:
# 
# 1. Data Analysis
# 2. Feature Engineering
# 3. Feature Selection
# 4. Model Building
# 
# **This is the notebook for step 4: Building the Final Machine Learning Model**
# 
# We will use the house price dataset available on [Kaggle.com](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). See below for more details.
# 
# ===================================================================================================
# 
# ## Predicting Sale Price of Houses
# 
# The aim of the project is to build a machine learning model to predict the sale price of homes based on different explanatory variables describing aspects of residential houses. 
# 
# ### Why is this important? 
# 
# Predicting house prices is useful to identify fruitful investments, or to determine whether the price advertised for a house is over or under-estimated.
# 
# ### What is the objective of the machine learning model?
# 
# We aim to minimise the difference between the real price and the price estimated by our model. We will evaluate model performance using the mean squared error (mse) and the root squared of the mean squared error (rmse).
# 
# ### How do I download the dataset?
# 
# To download the House Price dataset go this website:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# 
# Scroll down to the bottom of the page, and click on the link 'train.csv', and then click the 'download' blue button towards the right of the screen, to download the dataset. Rename the file as 'houseprice.csv' and save it to a directory of your choice.
# 
# **Note the following:**
# -  You need to be logged in to Kaggle in order to download the datasets.
# -  You need to accept the terms and conditions of the competition to download the dataset
# -  If you save the file to the same directory where you saved this jupyter notebook, then you can run the code as it is written here.
# 
# ====================================================================================================

# ## House Prices dataset: Model building
# 
# In the following cells, we will finally build our machine learning model, utilising the engineered data and the pre-selected features. 
# 
# 
# ### Setting the seed
# 
# It is important to note, that we are engineering variables and pre-processing data with the idea of deploying the model. Therefore, from now on, for each step that includes some element of randomness, it is extremely important that we **set the seed**. This way, we can obtain reproducibility between our research and our development code.
# 
# This is perhaps one of the most important lessons that you need to take away from this course: **Always set the seeds**.
# 
# Let's go ahead and load the dataset.

# In[1]:


# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt

# to build the model
from sklearn.linear_model import Lasso

# to evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# In[2]:


# load the train and test set with the engineered variables

# we built and saved these datasets in a previous notebook.
# If you haven't done so, go ahead and check the previous notebooks (step 2)
# to find out how to create these datasets

X_train = pd.read_csv('xtrain.csv')
X_test = pd.read_csv('xtest.csv')

X_train.head()


# In[3]:


# capture the target (remember that is log transformed)

y_train = X_train['SalePrice']
y_test = X_test['SalePrice']


# In[4]:


# load the pre-selected features
# ==============================

# we selected the features in the previous notebook (step 3)

# if you haven't done so, go ahead and visit the previous notebook
# to find out how to select the features

features = pd.read_csv('selected_features.csv')
features = features['0'].to_list() 

# We will add one additional feature to the ones we selected in the
# previous notebook: LotFrontage

# why?
#=====

# because it needs key feature engineering steps that we want to
# discuss further during the deployment part of the course. 

features = features + ['LotFrontage'] 

# display final feature set
features


# In[5]:


# reduce the train and test set to the selected features

X_train = X_train[features]
X_test = X_test[features]


# ### Regularised linear regression: Lasso
# 
# Remember to set the seed.

# In[6]:


# set up the model
# remember to set the random_state / seed

lin_model = Lasso(alpha=0.005, random_state=0)

# train the model

lin_model.fit(X_train, y_train)


# In[7]:


# evaluate the model:
# ====================

# remember that we log transformed the output (SalePrice)
# in our feature engineering notebook (step 2).

# In order to get the true performance of the Lasso
# we need to transform both the target and the predictions
# back to the original house prices values.

# We will evaluate performance using the mean squared error and
# the root of the mean squared error and r2

# make predictions for train set
pred = lin_model.predict(X_train)

# determine mse and rmse
print('train mse: {}'.format(int(
    mean_squared_error(np.exp(y_train), np.exp(pred)))))
print('train rmse: {}'.format(int(
    sqrt(mean_squared_error(np.exp(y_train), np.exp(pred))))))
print('train r2: {}'.format(
    r2_score(np.exp(y_train), np.exp(pred))))
print()

# make predictions for test set
pred = lin_model.predict(X_test)

# determine mse and rmse
print('test mse: {}'.format(int(
    mean_squared_error(np.exp(y_test), np.exp(pred)))))
print('test rmse: {}'.format(int(
    sqrt(mean_squared_error(np.exp(y_test), np.exp(pred))))))
print('test r2: {}'.format(
    r2_score(np.exp(y_test), np.exp(pred))))
print()

print('Average house price: ', int(np.exp(y_train).median()))


# In[8]:


# let's evaluate our predictions respect to the real sale price
plt.scatter(y_test, lin_model.predict(X_test))
plt.xlabel('True House Price')
plt.ylabel('Predicted House Price')
plt.title('Evaluation of Lasso Predictions')


# We can see that our model is doing a pretty good job at estimating house prices.

# In[9]:


# let's evaluate the distribution of the errors: 
# they should be fairly normally distributed

errors = y_test - lin_model.predict(X_test)
errors.hist(bins=30)


# The distribution of the errors follows quite closely a gaussian distribution. That suggests that our model is doing a good job as well.

# ### Feature importance

# In[10]:


# Finally, just for fun, let's look at the feature importance

importance = pd.Series(np.abs(lin_model.coef_.ravel()))
importance.index = features
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(18,6))
plt.ylabel('Lasso Coefficients')
plt.title('Feature Importance')


# And that is all! Now we have our entire pipeline ready for deployment. 
# 
# In the next video, we will summarise which steps from the pipeline we will deploy to production.

# In[ ]:




