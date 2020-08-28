# Setting Working Directory 

import os

# Printing Working Directory 
print(os.getcwd())

# Changing Working Directory
os.chdir("D:\SANJAY BDA\Online Profiles\LinkedIN\datasets")

# Printing Working Directory 
print(os.getcwd())

# Loading pandas library for loading dataset
import pandas as pd 

# Load data 
daata = pd.read_csv('classify risk.csv')

# first 5 record of the data
daata.head(5)

# information of the data
daata.info()

# gives quantiles(min, 1st quantile(25% of the data), 2nd quantile(50% of the data Median),
# 3rd quantile(75% of the data),max) and mean
daata.describe()

# data types of all variables
daata.dtypes

######################################## finding missing values ###############################
daata.isnull().any()

################################### Handling Categorical variables ###########################

# pd.getdummies : Creating dummy variables for categorical datatypes

from sklearn.preprocessing import OneHotEncoder
# it makes new columns of column as its category as if present then 1 otherwise 0

from sklearn.preprocessing import LabelEncoder
# it converts into 0,1,2.... as not make new columns 

lb = LabelEncoder()
#   mortgage = yes = 1 and no = 0
daata['mortgage'] = lb.fit_transform(daata['mortgage'])

# marital_status = others = 1 , single = 2 , married = 3
daata['marital_status'] = lb.fit_transform(daata['marital_status'])

# bad loss = 0 and good risk = 1
daata['risk'] = lb.fit_transform(daata['risk'])
daata

# import matplotlib.pyplot as plt

# plt.boxplot(daata['income'])

######################################## train / test spliting 

X = daata.iloc[:,0:5]
X
Y = daata.iloc[:,5]
Y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 

##############################################################################################
# DecisionTree
##############################################################################################
from sklearn.tree import DecisionTreeClassifier

model_1 = DecisionTreeClassifier()
model_1.fit(X_train,y_train)
y_pred_1 = model_1.predict(X_test)
y_pred_1

from sklearn.metrics import accuracy_score

accuracy_1 = accuracy_score(y_test,y_pred)
accuracy_1

from sklearn.metrics import confusion_matrix

confusion_matrix_2 = confusion_matrix(y_test,y_pred)
confusion_matrix_2

##############################################################################################
# Naive bayes
##############################################################################################
from sklearn.naive_bayes import GaussianNB

model_2 = GaussianNB()
model_2.fit(X_train,y_train)
y_pred_2 = model_2.predict(X_test)
y_pred_2

from sklearn.metrics import accuracy_score

accuracy_2 = accuracy_score(y_test,y_pred_2)
accuracy_2

from sklearn.metrics import confusion_matrix

confusion_matrix_2 = confusion_matrix(y_test,y_pred_2)
confusion_matrix_2

##############################################################################################
# Logistic Regression
##############################################################################################
from sklearn.linear_model import LogisticRegression

model_3 = LogisticRegression()
model_3.fit(X_train,y_train)
y_pred_3 = model_3.predict(X_test)
y_pred_3

from sklearn.metrics import accuracy_score

accuracy_3 = accuracy_score(y_test,y_pred_3)
accuracy_3

from sklearn.metrics import confusion_matrix

confusion_matrix_3 = confusion_matrix(y_test,y_pred_3)
confusion_matrix_3

##############################################################################################
# Support Vecotor Machine 
##############################################################################################
from sklearn.svm import SVC

model_4 = SVC()
model_4.fit(X_train,y_train)
y_pred_4 = model_4.predict(X_test)
y_pred_4

from sklearn.metrics import accuracy_score

accuracy_4 = accuracy_score(y_test,y_pred_4)
accuracy_4

from sklearn.metrics import confusion_matrix

confusion_matrix_4 = confusion_matrix(y_test,y_pred_4)
confusion_matrix_4

##############################################################################################
# Random Forest Classifier
##############################################################################################
from sklearn.ensemble import RandomForestClassifier

model_5 = RandomForestClassifier()
model_5.fit(X_train,y_train)
y_pred_5 = model_5.predict(X_test)
y_pred_5

from sklearn.metrics import accuracy_score

accuracy_5 = accuracy_score(y_test,y_pred_5)
accuracy_5

from sklearn.metrics import confusion_matrix

confusion_matrix_5 = confusion_matrix(y_test,y_pred_5)
confusion_matrix_5

##############################################################################################
# AdaBoost
##############################################################################################
from sklearn.ensemble import AdaBoostClassifier

model_6 = AdaBoostClassifier()
model_6.fit(X_train,y_train)
y_pred_6 = model_6.predict(X_test)
y_pred_6

from sklearn.metrics import accuracy_score

accuracy_6 = accuracy_score(y_test,y_pred_6)
accuracy_6

from sklearn.metrics import confusion_matrix

confusion_matrix_6 = confusion_matrix(y_test,y_pred_6)
confusion_matrix_6

##############################################################################################
# GradientBoost
##############################################################################################
from sklearn.ensemble import GradientBoostingClassifier

model_7 = GradientBoostingClassifier()
model_7.fit(X_train,y_train)
y_pred_7 = model_6.predict(X_test)
y_pred_7

from sklearn.metrics import accuracy_score

accuracy_7 = accuracy_score(y_test,y_pred_7)
accuracy_7

from sklearn.metrics import confusion_matrix

confusion_matrix_7 = confusion_matrix(y_test,y_pred_7)
confusion_matrix_7
