#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning Model Building Pipeline: Feature Engineering
# 
# In the following videos, we will take you through a practical example of each one of the steps in the Machine Learning model building pipeline, which we described in the previous lectures. There will be a notebook for each one of the Machine Learning Pipeline steps:
# 
# 1. Data Analysis
# 2. Feature Engineering
# 3. Feature Selection
# 4. Model Building
# 
# **This is the notebook for step 2: Feature Engineering**
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

# ## House Prices dataset: Feature Engineering
# 
# In the following cells, we will engineer / pre-process the variables of the House Price Dataset from Kaggle. We will engineer the variables so that we tackle:
# 
# 1. Missing values
# 2. Temporal variables
# 3. Non-Gaussian distributed variables
# 4. Categorical variables: remove rare labels
# 5. Categorical variables: convert strings to numbers
# 5. Standarise the values of the variables to the same range
# 
# ### Setting the seed
# 
# It is important to note that we are engineering variables and pre-processing data with the idea of deploying the model. Therefore, from now on, for each step that includes some element of randomness, it is extremely important that we **set the seed**. This way, we can obtain reproducibility between our research and our development code.
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

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)

import warnings
warnings.simplefilter(action='ignore')


# In[2]:


# load dataset
data = pd.read_csv('houseprice.csv')
print(data.shape)
data.head()


# ## Separate dataset into train and test
# 
# Before beginning to engineer our features, it is important to separate our data intro training and testing set. When we engineer features, some techniques learn parameters from data. It is important to learn this parameters only from the train set. This is to avoid over-fitting. 
# 
# **Separating the data into train and test involves randomness, therefore, we need to set the seed.**

# In[3]:


# Let's separate into train and test set
# Remember to set the seed (random_state for this sklearn function)

X_train, X_test, y_train, y_test = train_test_split(data,
                                                    data['SalePrice'],
                                                    test_size=0.1,
                                                    # we are setting the seed here:
                                                    random_state=0)  

X_train.shape, X_test.shape


# ## Missing values
# 
# ### Categorical variables
# For categorical variables, we will replace missing values with the string "missing".

# In[4]:


# make a list of the categorical variables that contain missing values

vars_with_na = [
    var for var in data.columns
    if X_train[var].isnull().sum() > 0 and X_train[var].dtypes == 'O'
]

# print percentage of missing values per variable
X_train[vars_with_na].isnull().mean()


# In[5]:


# replace missing values with new label: "Missing"

X_train[vars_with_na] = X_train[vars_with_na].fillna('Missing')
X_test[vars_with_na] = X_test[vars_with_na].fillna('Missing')


# In[6]:


# check that we have no missing information in the engineered variables
X_train[vars_with_na].isnull().sum()


# In[7]:


# check that test set does not contain null values in the engineered variables
[var for var in vars_with_na if X_test[var].isnull().sum() > 0]


# ### Numerical variables
# 
# To engineer missing values in numerical variables, we will:
# 
# - add a binary missing value indicator variable
# - and then replace the missing values in the original variable with the mode
# 

# In[8]:


# make a list with the numerical variables that contain missing values
vars_with_na = [
    var for var in data.columns
    if X_train[var].isnull().sum() > 0 and X_train[var].dtypes != 'O'
]

# print percentage of missing values per variable
X_train[vars_with_na].isnull().mean()


# In[9]:


# replace engineer missing values as we described above

for var in vars_with_na:

    # calculate the mode using the train set
    mode_val = X_train[var].mode()[0]

    # add binary missing indicator (in train and test)
    X_train[var+'_na'] = np.where(X_train[var].isnull(), 1, 0)
    X_test[var+'_na'] = np.where(X_test[var].isnull(), 1, 0)

    # replace missing values by the mode
    # (in train and test)
    X_train[var] = X_train[var].fillna(mode_val)
    X_test[var] = X_test[var].fillna(mode_val)

# check that we have no more missing values in the engineered variables
X_train[vars_with_na].isnull().sum()


# In[10]:


# check that test set does not contain null values in the engineered variables

[vr for var in vars_with_na if X_test[var].isnull().sum() > 0]


# In[11]:


# check the binary missing indicator variables

X_train[['LotFrontage_na', 'MasVnrArea_na', 'GarageYrBlt_na']].head()


# ## Temporal variables
# 
# ### Capture elapsed time
# 
# We learned in the previous Jupyter notebook, that there are 4 variables that refer to the years in which the house or the garage were built or remodeled. We will capture the time elapsed between those variables and the year in which the house was sold:

# In[12]:


def elapsed_years(df, var):
    # capture difference between the year variable
    # and the year in which the house was sold
    df[var] = df['YrSold'] - df[var]
    return df


# In[13]:


for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    X_train = elapsed_years(X_train, var)
    X_test = elapsed_years(X_test, var)


# ## Numerical variable transformation
# 
# In the previous Jupyter notebook, we observed that the numerical variables are not normally distributed.
# 
# We will log transform the positive numerical variables in order to get a more Gaussian-like distribution. This tends to help Linear machine learning models. 

# In[14]:


for var in ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']:
    X_train[var] = np.log(X_train[var])
    X_test[var] = np.log(X_test[var])


# In[15]:


# check that test set does not contain null values in the engineered variables
[var for var in ['LotFrontage', 'LotArea', '1stFlrSF',
                 'GrLivArea', 'SalePrice'] if X_test[var].isnull().sum() > 0]


# In[16]:


# same for train set
[var for var in ['LotFrontage', 'LotArea', '1stFlrSF',
                 'GrLivArea', 'SalePrice'] if X_train[var].isnull().sum() > 0]


# ## Categorical variables
# 
# ### Removing rare labels
# 
# First, we will group those categories within variables that are present in less than 1% of the observations. That is, all values of categorical variables that are shared by less than 1% of houses, well be replaced by the string "Rare".
# 
# To learn more about how to handle categorical variables visit our course [Feature Engineering for Machine Learning](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=UDEMY2018) in Udemy.

# In[17]:


# let's capture the categorical variables in a list

cat_vars = [var for var in X_train.columns if X_train[var].dtype == 'O']


# In[18]:


def find_frequent_labels(df, var, rare_perc):
    
    # function finds the labels that are shared by more than
    # a certain % of the houses in the dataset

    df = df.copy()

    tmp = df.groupby(var)['SalePrice'].count() / len(df)

    return tmp[tmp > rare_perc].index


for var in cat_vars:
    
    # find the frequent categories
    frequent_ls = find_frequent_labels(X_train, var, 0.01)
    
    # replace rare categories by the string "Rare"
    X_train[var] = np.where(X_train[var].isin(
        frequent_ls), X_train[var], 'Rare')
    
    X_test[var] = np.where(X_test[var].isin(
        frequent_ls), X_test[var], 'Rare')


# ### Encoding of categorical variables
# 
# Next, we need to transform the strings of the categorical variables into numbers. We will do it so that we capture the monotonic relationship between the label and the target.
# 
# To learn more about how to encode categorical variables visit our course [Feature Engineering for Machine Learning](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=UDEMY2018) in Udemy.

# In[19]:


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


# In[20]:


for var in cat_vars:
    replace_categories(X_train, X_test, var, 'SalePrice')


# In[21]:


# check absence of na in the train set
[var for var in X_train.columns if X_train[var].isnull().sum() > 0]


# In[22]:


# check absence of na in the test set
[var for var in X_test.columns if X_test[var].isnull().sum() > 0]


# In[23]:


# let me show you what I mean by monotonic relationship
# between labels and target

def analyse_vars(df, var):
    
    # function plots median house sale price per encoded
    # category
    
    df = df.copy()
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.ylabel('SalePrice')
    plt.show()
    
for var in cat_vars:
    analyse_vars(X_train, var)


# The monotonic relationship is particularly clear for the variables MSZoning, Neighborhood, and ExterQual. Note how, the higher the integer that now represents the category, the higher the mean house sale price.
# 
# (remember that the target is log-transformed, that is why the differences seem so small).

# ## Feature Scaling
# 
# For use in linear models, features need to be either scaled or normalised. In the next section, I will scale features to the minimum and maximum values:

# In[24]:


# capture all variables in a list
# except the target and the ID

train_vars = [var for var in X_train.columns if var not in ['Id', 'SalePrice']]

# count number of variables
len(train_vars)


# In[25]:


# create scaler
scaler = MinMaxScaler()

#  fit  the scaler to the train set
scaler.fit(X_train[train_vars]) 

# transform the train and test set
X_train[train_vars] = scaler.transform(X_train[train_vars])

X_test[train_vars] = scaler.transform(X_test[train_vars])


# In[26]:


X_train.head()


# In[27]:


# let's now save the train and test sets for the next notebook!

X_train.to_csv('xtrain.csv', index=False)
X_test.to_csv('xtest.csv', index=False)


# That concludes the feature engineering section for this dataset.
# 
# **Remember: the aim of this course is to show you how to put models in production. We deliberately kept the feature engineering pipeline, yet included many of the traditional engineering steps, to give you a full flavour of building and deploying a machine learning model pipeline** as we will see in the coming sections of the course.

# That is all for this notebook. We hope you enjoyed it and see you in the next one!
