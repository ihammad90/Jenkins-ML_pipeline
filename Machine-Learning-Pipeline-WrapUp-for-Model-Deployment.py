#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning Pipeline: Wrapping up for Deployment
# 
# 
# In the previous notebooks, we worked through the typical Machine Learning pipeline steps to build a regression model that allows us to predict house prices. Briefly, we transformed variables in the dataset to make them suitable for use in a Regression model, then we selected the most predictive variables and finally we trained our model.
# 
# Now, we want to deploy our model. We want to create an API, which we can call with new data, with new characteristics about houses, to get an estimate of the SalePrice. In order to do so, we need to write code in a very specific way. We will show you how to write production code in the next sections.
# 
# Here, we will summarise the key pieces of code, that we need to take forward for this particular project, to put our model in production.
# 
# Let's go ahead and get started.

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

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# to build the models
from sklearn.linear_model import Lasso

# to evaluate the models
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# to persist the model and the scaler
import joblib

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)

import warnings
warnings.simplefilter(action='ignore')


# ## Load data
# 
# We need the training data to train our model in the production environment. 

# In[2]:


# load dataset
data = pd.read_csv('houseprice.csv')
print(data.shape)
data.head()


# ## Separate dataset into train and test

# In[3]:


X_train, X_test, y_train, y_test = train_test_split(
    data,
    data['SalePrice'],
    test_size=0.1,
    # we are setting the seed here
    random_state=0)

X_train.shape, X_test.shape


# In[4]:


X_train.head()


# ## Selected features

# In[5]:


# load selected features
features = pd.read_csv('selected_features.csv')

# Added the extra feature, LotFrontage
features = features['0'].to_list() + ['LotFrontage']

print('Number of features: ', len(features))


# ## Engineer missing values
# 
# ### Categorical variables
# 
# For categorical variables, we will replace missing values with the string "missing".

# In[6]:


# make a list of the categorical variables that contain missing values

vars_with_na = [
    var for var in features
    if X_train[var].isnull().sum() > 0 and X_train[var].dtypes == 'O'
]

# display categorical variables that we will engineer:
vars_with_na


# Note that we have much less categorical variables with missing values than in our original dataset. But we still use categorical variables with NA for the final model, so we need to include this piece of feature engineering logic in the deployment pipeline. 

# In[7]:


# I bring forward the code used in the feature engineering notebook:
# (step 2)

X_train[vars_with_na] = X_train[vars_with_na].fillna('Missing')
X_test[vars_with_na] = X_test[vars_with_na].fillna('Missing')

# check that we have no missing information in the engineered variables
X_train[vars_with_na].isnull().sum()


# ### Numerical variables
# 
# To engineer missing values in numerical variables, we will:
# 
# - add a binary missing value indicator variable
# - and then replace the missing values in the original variable with the mode
# 

# In[8]:


# make a list of the numerical variables that contain missing values:

vars_with_na = [
    var for var in features
    if X_train[var].isnull().sum() > 0 and X_train[var].dtypes != 'O'
]

# display numerical variables with NA
vars_with_na


# In[9]:


# I bring forward the code used in the feature engineering notebook
# with minor adjustments (step 2):

var = 'LotFrontage'

# calculate the mode
mode_val = X_train[var].mode()[0]
print('mode of LotFrontage: {}'.format(mode_val))

# replace missing values by the mode
# (in train and test)
X_train[var] = X_train[var].fillna(mode_val)
X_test[var] = X_test[var].fillna(mode_val)


# ## Temporal variables
# 
# One of our temporal variables was selected to be used in the final model: 'YearRemodAdd'
# 
# So we need to deploy the bit of code that creates it.

# In[10]:


# create the temporal var "elapsed years"

# I bring this bit of code forward from the notebook on feature
# engineering (step 2)

def elapsed_years(df, var):
    # capture difference between year variable
    # and year in which the house was sold
    
    df[var] = df['YrSold'] - df[var]
    
    return df


# In[11]:


X_train = elapsed_years(X_train, 'YearRemodAdd')
X_test = elapsed_years(X_test, 'YearRemodAdd')


# ### Numerical variable transformation

# In[12]:


# we apply the logarithmic function to the variables that
# were selected (and the target):

for var in ['LotFrontage', '1stFlrSF', 'GrLivArea', 'SalePrice']:
    X_train[var] = np.log(X_train[var])
    X_test[var] = np.log(X_test[var])


# ## Categorical variables
# 
# ### Group rare labels

# In[13]:


# let's capture the categorical variables first

cat_vars = [var for var in features if X_train[var].dtype == 'O']

cat_vars


# In[14]:


# bringing thise from the notebook on feature engineering (step 2):

def find_frequent_labels(df, var, rare_perc):
    
    # function finds the labels that are shared by more than
    # a certain % of the houses in the dataset

    df = df.copy()

    tmp = df.groupby(var)['SalePrice'].count() / len(df)

    return tmp[tmp > rare_perc].index


for var in cat_vars:
    
    # find the frequent categories
    frequent_ls = find_frequent_labels(X_train, var, 0.01)
    print(var)
    print(frequent_ls)
    print()
    
    # replace rare categories by the string "Rare"
    X_train[var] = np.where(X_train[var].isin(
        frequent_ls), X_train[var], 'Rare')
    
    X_test[var] = np.where(X_test[var].isin(
        frequent_ls), X_test[var], 'Rare')


# ### Encoding of categorical variables
# 

# In[15]:


# this function will assign discrete values to the strings of the variables,
# so that the smaller value corresponds to the category that shows the smaller
# mean house sale price


def replace_categories(train, test, var, target):

    # order the categories in a variable from that with the lowest
    # house sale price, to that with the highest
    ordered_labels = train.groupby([var])[target].mean().sort_values().index

    # create a dictionary of ordered categories to integer values
    ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}

    # use the dictionary to replace the categorical strings by integers
    train[var] = train[var].map(ordinal_label)
    test[var] = test[var].map(ordinal_label)
    
    print(var)
    print(ordinal_label)
    print()


# In[16]:


for var in cat_vars:
    replace_categories(X_train, X_test, var, 'SalePrice')


# In[17]:


# check absence of na
[var for var in features if X_train[var].isnull().sum() > 0]


# In[18]:


# check absence of na
[var for var in features if X_test[var].isnull().sum() > 0]


# ### Feature Scaling
# 
# For use in linear models, features need to be either scaled or normalised. In the next section, I will scale features between the min and max values:

# In[19]:


# capture the target
y_train = X_train['SalePrice']
y_test = X_test['SalePrice']


# In[20]:


# set up scaler
scaler = MinMaxScaler()

# train scaler
scaler.fit(X_train[features])


# In[21]:


# explore maximum values of variables
scaler.data_max_


# In[22]:


# explore minimum values of variables
scaler.data_min_


# In[23]:


# transform the train and test set, and add on the Id and SalePrice variables
X_train = scaler.transform(X_train[features])
X_test = scaler.transform(X_test[features])


# ## Train the Linear Regression: Lasso

# In[24]:


# set up the model
# remember to set the random_state / seed

lin_model = Lasso(alpha=0.005, random_state=0)

# train the model
lin_model.fit(X_train, y_train)

# we persist the model for future use
joblib.dump(lin_model, 'lasso_regression.pkl')


# In[25]:


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


# That is all for this notebook. And that is all for this section too.
# 
# **In the next section, we will show you how to productionise this code for model deployment**.

# In[ ]:




