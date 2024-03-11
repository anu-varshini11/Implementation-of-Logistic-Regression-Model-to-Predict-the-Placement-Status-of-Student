# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preprocessing: Cleanse data, handle missing values, encode categorical variables.
2. Model Training: Fit logistic regression model on preprocessed data.
3. Model Evaluation: Assess model performance using metrics like accuracy, precision, recall.
4. Prediction: Predict placement status for new student data using trained model.

## Program:
```python3
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ANU VARSHINI M B
RegisterNumber: 212223240010

import pandas as pd
data=pd.read_csv("C:/Users/admin/OneDrive/Documents/INTRO TO ML/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data.head()

data1.isnull()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![ml1](https://github.com/anu-varshini11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138969827/a5c3d163-7f8d-45e4-b651-9ff2adccdf5f)
![ml2](https://github.com/anu-varshini11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138969827/c6dd32d6-c425-4403-bc3e-00356d3c3205)
![ml3](https://github.com/anu-varshini11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138969827/16c18f32-7e2c-4c07-b0ac-eb99936b8806)
![ml4](https://github.com/anu-varshini11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138969827/cabbda4f-c5e6-4c86-9df2-80bfb22df2dd)
![ml5](https://github.com/anu-varshini11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138969827/eb61ba81-2e6b-4db0-8709-853111d38fea)
![ml6](https://github.com/anu-varshini11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138969827/5c672ced-749a-433c-8ceb-b087fcfabf4e)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
