#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning Model Building Pipeline: Feature Selection
# 
# In the following videos, we will take you through a practical example of each one of the steps in the Machine Learning model building pipeline, which we described in the previous lectures. There will be a notebook for each one of the Machine Learning Pipeline steps:
# 
# 1. Data Analysis
# 2. Feature Engineering
# 3. Feature Selection
# 4. Model Building
# 
# **This is the notebook for step 3: Feature Selection**
# 
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

# ## House Prices dataset: Feature Selection
# 
# In the following cells, we will select a group of variables, the most predictive ones, to build our machine learning model. 
# 
# ### Why do we select variables?
# 
# - For production: Fewer variables mean smaller client input requirements (e.g. customers filling out a form on a website or mobile app), and hence less code for error handling. This reduces the chances of introducing bugs.
# 
# - For model performance: Fewer variables mean simpler, more interpretable, better generalizing models
# 
# 
# **We will select variables using the Lasso regression: Lasso has the property of setting the coefficient of non-informative variables to zero. This way we can identify those variables and remove them from our final model.**
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

# to build the models
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# In[2]:


# load the train and test set with the engineered variables

# we built and saved these datasets in the previous lecture.
# If you haven't done so, go ahead and check the previous notebook
# to find out how to create these datasets

X_train = pd.read_csv('xtrain.csv')
X_test = pd.read_csv('xtest.csv')

X_train.head()


# In[3]:


# capture the target (remember that the target is log transformed)
y_train = X_train['SalePrice']
y_test = X_test['SalePrice']

# drop unnecessary variables from our training and testing sets
X_train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
X_test.drop(['Id', 'SalePrice'], axis=1, inplace=True)


# ### Feature Selection
# 
# Let's go ahead and select a subset of the most predictive features. There is an element of randomness in the Lasso regression, so remember to set the seed.

# In[4]:


# We will do the model fitting and feature selection
# altogether in a few lines of code

# first, we specify the Lasso Regression model, and we
# select a suitable alpha (equivalent of penalty).
# The bigger the alpha the less features that will be selected.

# Then we use the selectFromModel object from sklearn, which
# will select automatically the features which coefficients are non-zero

# remember to set the seed, the random state in this function
sel_ = SelectFromModel(Lasso(alpha=0.005, random_state=0))

# train Lasso model and select features
sel_.fit(X_train, y_train)


# In[5]:


# let's visualise those features that were selected.
# (selected features marked with True)

sel_.get_support()


# In[6]:


# let's print the number of total and selected features

# this is how we can make a list of the selected features
selected_feats = X_train.columns[(sel_.get_support())]

# let's print some stats
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feats)))
print('features with coefficients shrank to zero: {}'.format(
    np.sum(sel_.estimator_.coef_ == 0)))


# In[7]:


# print the selected features
selected_feats


# ### Identify the selected variables

# In[8]:


# this is an alternative way of identifying the selected features
# based on the non-zero regularisation coefficients:

selected_feats = X_train.columns[(sel_.estimator_.coef_ != 0).ravel().tolist()]

selected_feats


# In[9]:


pd.Series(selected_feats).to_csv('selected_features.csv', index=False)


# That is all for this notebook. In the next video, we will go ahead and build the final model using the selected features. See you then!
