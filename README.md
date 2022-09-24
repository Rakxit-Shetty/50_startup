# 50_startup
Regression Model on Startup profit
Introduction
We will use the features at our disposal to forecast the profit from the startup's dataset. For this problem statement, we will use the 50 startups dataset and the idea of multiple linear regression to forecast the profitability of startup's profit.
A number of explanatory variables are combined in a statistical process called multiple linear regression (MLR), also referred to as multiple regression.
We will essentially have multiple features, such as f1, f2, f3, f4, and our output feature(target), f5.
About the 50_startups dataset
This particular dataset holds data from 50 startups in New York, California, and Florida. The features in this dataset are R&D spending, Administration Spending, Marketing Spending, and location features, while the target variable is Profit.
 R&D spending: The amount which startups are spending on Research and development.
Administration spending: The amount which startups are spending on the admin panel.
Marketing spending: The amount which startups are spending on marketing strategies.
State: To which state that particular startup belongs.
Profit: How much profit that particular startup is making.
you can find the dataset here.
How can this model help here?
This machine learning model will be quite helpful in such a situation where we need to find a profit based on how much we are spending in the market and for the market. In a nutshell, this machine learning model will help to find out the profit based on the amount which we spend from the 50 startups dataset.
LET US START:
Reading dataset
The majority of the datasets are in CSV files, and we utilize the pandas package to read these files.
import pandas as pd
df = pd.read_csv('D:/Rakshith/mydataset/50_Startups.csv')
#show the first 5 data that has been read into df
df.head()
Before acting on any datasets we have to understand the features, dimension of the datasets and basics of data Preparation.
Listing Missing Values
Let's examine the proportion of missing values in the dataset:
df.isnull().sum()
There are no null values in the dataset, hence we can move forward.
Now, we can find the correlation between the columns by using
df.corr()
We can see that R&D spending, Administration Spending, Marketing Spending columns have a direct relationship with the profit, which is our target variable.

Spliting Dataset in Dependent & Independent Variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
Data Preprocessing
Preprocessed data is more important than the most potent algorithms that machine learning models trained on faulty data could actually hinder the analysis you're trying to do by producing "junk" results. It is not always the case that we come across the clean and prepared data when developing a machine learning model. Additionally, any time you work with data, you must clean it up and format it.
We must handle any categorical values in the dataset included in the categorical column State in our dataset, hence we will utilize the LabelEncoder
handle categorical variable
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X1 = pd.DataFrame(X)
X1.head()
Splitting Data
Now, we have to split the data into training and testing parts, for that we use the scikit-learn train_test_split() function.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=50)
The dataset is now split into 20% test size data and 80% training size data with the randomness of 50  rows 
Applying the Regression model
Since we have multiple dependent variables, we have to go with Multiple linear regression. There is total 5 features in the dataset, in which basically profit is our dependent feature or target feature, and the rest of them are our independent features.
Now, we have to import linear regression from the scikit-learn library
from sklearn.linear_model import LinearRegression
Creating an object of LinearRegression class
MLR = LinearRegression()
Fitting the training data
MLR.fit(x_train,y_train)
Finally, if we run this, our model will be ready. Since we already have the x test data, we can use this data for the prediction of profit.
y_prediction =  LR.predict(x_test)
y_pred=y_prediction
Now, we have to compare the y_prediction values with the original values because we have to calculate the accuracy of our model,
Model evaluation
R2 score: R2 score - R squared score. It is one of the statistical approaches by which we can find the variance or the spread of the target and feature data.
from sklearn.metrics import r2_score

r2Score = r2_score(y_pred, y_test)
print("R2 score of model is :" ,r2Score*100)
>>>R2 score of model is: 1013276.1777119015
MSE: MSE - Mean Squared Error. By using this approach we can find that how much the regression best fit line is close to all the residual.
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_pred, y_test)
print("Mean Squarred Error is :" ,mse*100)
>>>Mean Squared Error is: 10267286123.18441
RMSE: RMSE - Root Mean Squared Error. This is similar to the Mean squared error(MSE) approach, the only difference is that here we find the root of the mean squared error i.e. root of the Mean squared error is equal to Root Mean Squared Error. The reason behind finding the root is to find the more close residual to the values found by mean squared error.
import numpy as np
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
print("Root Mean Squarred Error is : ",rmse*100)
>>>Root Mean Squarred Error is :  1013276.1777119015
MAE: MAE - Mean Absolute Error. By using this approach we can find the difference between the actual values and predicted values but that difference is absolute i.e. the difference is positive.
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_pred,y_test)
print("Mean Absolute Error is :" ,mae)
>>>Mean Absolute Error is : 8536.718825231252
Conclusion
So, the mean absolute error is 8536.718825231252. Therefore our predicted value can be 8536.718825231252 units more or less than the actual value.
