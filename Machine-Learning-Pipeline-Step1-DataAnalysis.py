#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning Model Building Pipeline: Data Analysis
# 
# In the following videos, we will take you through a practical example of each one of the steps in the Machine Learning model building pipeline, which we described in the previous lectures. There will be a notebook for each one of the Machine Learning Pipeline steps:
# 
# 1. Data Analysis
# 2. Feature Engineering
# 3. Feature Selection
# 4. Model Building
# 
# **This is the notebook for step 1: Data Analysis**
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

# ## House Prices dataset: Data Analysis
# 
# In the following cells, we will analyse the variables of the House Price Dataset from Kaggle. We will take you through the different aspects of the analysis of the variables, and introduce you to the meaning of each of the variables in the dataset as well. If you want to know more about this dataset, visit [Kaggle.com](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
# 
# Let's go ahead and load the dataset.

# In[1]:


# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt

# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)


# In[2]:


# load dataset
data = pd.read_csv('houseprice.csv')

# rows and columns of the data
print(data.shape)

# visualise the dataset
data.head()


# The house price dataset contains 1460 rows, i.e., houses, and 81 columns, i.e., variables. 
# 
# **We will analyse the dataset to identify:**
# 
# 1. Missing values
# 2. Numerical variables
# 3. Distribution of the numerical variables
# 4. Outliers
# 5. Categorical variables
# 6. Cardinality of the categorical variables
# 7. Potential relationship between the variables and the target: SalePrice

# ### Missing values
# 
# Let's go ahead and find out which variables of the dataset contain missing values.

# In[3]:


# make a list of the variables that contain missing values
vars_with_na = [var for var in data.columns if data[var].isnull().sum() > 0]

# determine percentage of missing values
data[vars_with_na].isnull().mean()


# Our dataset contains a few variables with missing values. We need to account for this in our following notebook / video, where we will engineer the variables for use in Machine Learning Models.

# #### Relationship between values being missing and Sale Price
# 
# Let's evaluate the price of the house in those observations where the information is missing, for each variable.

# In[4]:


def analyse_na_value(df, var):

    df = df.copy()

    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    df[var] = np.where(df[var].isnull(), 1, 0)

    # let's compare the median SalePrice in the observations where data is missing
    # vs the observations where a value is available

    df.groupby(var)['SalePrice'].median().plot.bar()

    plt.title(var)
    plt.show()


# let's run the function on each variable with missing data
for var in vars_with_na:
    analyse_na_value(data, var)


# The average Sale Price in houses where the information is missing, differs from the average Sale Price in houses where information exists. 
# 
# We will capture this information when we engineer the variables in our next lecture / video.

# ### Numerical variables
# 
# Let's go ahead and find out what numerical variables we have in the dataset

# In[5]:


# make list of numerical variables
num_vars = [var for var in data.columns if data[var].dtypes != 'O']

print('Number of numerical variables: ', len(num_vars))

# visualise the numerical variables
data[num_vars].head()


# From the above view of the dataset, we notice the variable Id, which is an indicator of the house. We will not use this variable to make our predictions, as there is one different value of the variable per each row, i.e., each house in the dataset. See below:

# In[6]:


print('Number of House Id labels: ', len(data.Id.unique()))
print('Number of Houses in the Dataset: ', len(data))


# #### Temporal variables
# 
# We have 4 year variables in the dataset:
# 
# - YearBuilt: year in which the house was built
# - YearRemodAdd: year in which the house was remodeled
# - GarageYrBlt: year in which a garage was built
# - YrSold: year in which the house was sold
# 
# We generally don't use date variables in their raw format. Instead, we extract information from them. For example, we can capture the difference in years between the year the house was built and the year the house was sold.

# In[7]:


# list of variables that contain year information

year_vars = [var for var in num_vars if 'Yr' in var or 'Year' in var]

year_vars


# In[8]:


# let's explore the values of these temporal variables

for var in year_vars:
    print(var, data[var].unique())
    print()


# As expected, the values are years.
# 
# We can explore the evolution of the sale price with the years in which the house was sold:

# In[9]:


data.groupby('YrSold')['SalePrice'].median().plot()
plt.ylabel('Median House Price')
plt.title('Change in House price with the years')


# There has been a drop in the value of the houses. That is unusual, in real life, house prices typically go up as years go by.
# 
# 
# Let's go ahead and explore whether there is a relationship between the year variables and SalePrice. For this, we will capture the elapsed years between the Year variables and the year in which the house was sold:

# In[10]:


# let's explore the relationship between the year variables
# and the house price in a bit of more detail:

def analyse_year_vars(df, var):
    df = df.copy()
    
    # capture difference between year variable and year
    # in which the house was sold
    df[var] = df['YrSold'] - df[var]
    
    plt.scatter(df[var], df['SalePrice'])
    plt.ylabel('SalePrice')
    plt.xlabel(var)
    plt.show()
    
    
for var in year_vars:
    if var !='YrSold':
        analyse_year_vars(data, var)
    


# We see that there is a tendency to a decrease in price, with older features. In other words, the longer the time between the house was built or remodeled and sale date, the lower the sale Price. 
# 
# Which makes sense, cause this means that the house will have an older look, and potentially needs repairs.

# #### Discrete variables
# 
# Let's go ahead and find which variables are discrete, i.e., show a finite number of values

# In[11]:


#  let's male a list of discrete variables
discrete_vars = [var for var in num_vars if len(
    data[var].unique()) < 20 and var not in year_vars+['Id']]


print('Number of discrete variables: ', len(discrete_vars))


# In[12]:


# let's visualise the discrete variables

data[discrete_vars].head()


# These discrete variables tend to be qualifications or grading scales, or refer to the number of rooms, or units.
# 
# Let's go ahead and analyse their contribution to the house price.

# In[13]:


def analyse_discrete(df, var):
    df = df.copy()
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.ylabel('Median SalePrice')
    plt.show()
    
for var in discrete_vars:
    analyse_discrete(data, var)


# There tend to be a relationship between the variables values and the SalePrice, but this relationship is not always monotonic. 
# 
# For example, for OverallQual, there is a monotonic relationship: the higher the quality, the higher the SalePrice.  
# 
# However, for OverallCond, the relationship is not monotonic. Clearly, some Condition grades, like 5, correlate with higher sale prices, but higher values do not necessarily do so. We need to be careful on how we engineer these variables to extract maximum value for a linear model.
# 
# There are ways to re-arrange the order of the discrete values of a variable, to create a monotonic relationship between the variable and the target. However, for the purpose of this course, we will not do that, to keep feature engineering simple. If you want to learn more about how to engineer features, visit our course [Feature Engineering for Machine Learning](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=UDEMY2018) in Udemy.

# #### Continuous variables
# 
# Let's go ahead and find the distribution of the continuous variables. We will consider continuous variables to all those that are not temporal or discrete variables in our dataset.

# In[14]:


# make list of continuous variables
cont_vars = [
    var for var in num_vars if var not in discrete_vars+year_vars+['Id']]

print('Number of continuous variables: ', len(cont_vars))


# In[15]:


# let's visualise the continuous variables

data[cont_vars].head()


# In[16]:


# Let's go ahead and analyse the distributions of these variables


def analyse_continuous(df, var):
    df = df.copy()
    df[var].hist(bins=30)
    plt.ylabel('Number of houses')
    plt.xlabel(var)
    plt.title(var)
    plt.show()


for var in cont_vars:
    analyse_continuous(data, var)


# The variables are not normally distributed, including the target variable 'SalePrice'. 
# 
# To maximise performance of linear models, we need to account for non-Gaussian distributions. We will transform our variables in the next lecture / video, during our feature engineering step.
# 
# Let's evaluate if a logarithmic transformation of the variables returns values that follow a normal distribution:

# In[17]:


# Let's go ahead and analyse the distributions of these variables
# after applying a logarithmic transformation


def analyse_transformed_continuous(df, var):
    df = df.copy()

    # log does not take 0 or negative values, so let's be
    # careful and skip those variables
    if any(data[var] <= 0):
        pass
    else:
        # log transform the variable
        df[var] = np.log(df[var])
        df[var].hist(bins=30)
        plt.ylabel('Number of houses')
        plt.xlabel(var)
        plt.title(var)
        plt.show()


for var in cont_vars:
    analyse_transformed_continuous(data, var)


# We get a better spread of the values for most variables when we use the logarithmic transformation. This engineering step will most likely add performance value to our final model.

# In[18]:


# let's explore the relationship between the house price and
# the transformed variables with more detail:


def transform_analyse_continuous(df, var):
    df = df.copy()

    # log does not take negative values, so let's be careful and skip those variables
    if any(data[var] <= 0):
        pass
    else:
        # log transform the variable
        df[var] = np.log(df[var])
        
        # log transform the target (remember it was also skewed)
        df['SalePrice'] = np.log(df['SalePrice'])
        
        # plot
        plt.scatter(df[var], df['SalePrice'])
        plt.ylabel('SalePrice')
        plt.xlabel(var)
        plt.show()


for var in cont_vars:
    if var != 'SalePrice':
        transform_analyse_continuous(data, var)


# From the previous plots, we observe some monotonic associations between SalePrice and the variables to which we applied the log transformation, for example 'GrLivArea'.

# #### Outliers
# 
# Extreme values may affect the performance of a linear model. Let's find out if we have any in our variables.

# In[19]:


# let's make boxplots to visualise outliers in the continuous variables


def find_outliers(df, var):
    df = df.copy()

    # log does not take negative values, so let's be
    # careful and skip those variables
    if any(data[var] <= 0):
        pass
    else:
        df[var] = np.log(df[var])
        df.boxplot(column=var)
        plt.title(var)
        plt.ylabel(var)
        plt.show()


for var in cont_vars:
    find_outliers(data, var)


# The majority of the continuous variables seem to contain outliers. Outliers tend to affect the performance of linear model. So it is worth spending some time understanding if removing outliers will add performance value to our  final machine learning model.
# 
# The purpose of this course is however to teach you how to put your models in production. Therefore, we will not spend more time looking at how best to remove outliers, and we will rather deploy a simpler model.
# 
# However, if you want to learn more about the value of removing outliers, visit our course [Feature Engineering for Machine Learning](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=UDEMY2018).
# 
# The same is true for variable transformation. There are multiple ways to improve the spread of the variable over a wider range of values. You can learn more about it in our course [Feature Engineering for Machine Learning](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=UDEMY2018).

# ### Categorical variables
# 
# Let's go ahead and analyse the categorical variables present in the dataset.

# In[20]:


# capture categorical variables in a list
cat_vars = [var for var in data.columns if data[var].dtypes == 'O']

print('Number of categorical variables: ', len(cat_vars))


# In[21]:


# let's visualise the values of the categorical variables
data[cat_vars].head()


# #### Number of labels: cardinality
# 
# Let's evaluate how many different categories are present in each of the variables.

# In[22]:


data[cat_vars].nunique()


# All the categorical variables show low cardinality, this means that they have only few different labels. That is good as we won't need to tackle cardinality during our feature engineering lecture.
# 
# #### Rare labels:
# 
# Let's go ahead and investigate now if there are labels that are present only in a small number of houses:

# In[23]:


def analyse_rare_labels(df, var, rare_perc):
    df = df.copy()

    # determine the % of observations per category
    tmp = df.groupby(var)['SalePrice'].count() / len(df)

    # return categories that are rare
    return tmp[tmp < rare_perc]

# print categories that are present in less than
# 1 % of the observations


for var in cat_vars:
    print(analyse_rare_labels(data, var, 0.01))
    print()


# Some of the categorical variables show multiple labels that are present in less than 1% of the houses. We will engineer these variables in our next video. Labels that are under-represented in the dataset tend to cause over-fitting of machine learning models. That is why we want to remove them.
# 
# Finally, we want to explore the relationship between the categories of the different variables and the house sale price:

# In[24]:


for var in cat_vars:
    # we can re-use the function to determine median
    # sale price, that we created for discrete variables

    analyse_discrete(data, var)


# Clearly, the categories give information on the SalePrice, as different categories show different median sale prices.
# 
# In the next video, we will transform these strings / labels into numbers, so that we capture this information and transform it into a monotonic relationship between the category and the house price.

# **Disclaimer:**
# 
# The data exploration shown in this notebook by no means wants to be an exhaustive data exploration. There is certainly more to be done to understand the nature of this data and the relationship of these variables with the target, SalePrice.
# 
# However, we hope that through this notebook we gave you both a flavour of what data analysis looks like, and set the bases for the coming steps in the machine learning model building pipeline. Through data exploration, we decide which feature engineering techniques we will apply to our variables.

# That is all for this lecture / notebook. I hope you enjoyed it, and see you in the next one!
