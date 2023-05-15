# EXPERIMENT 07: IMPLEMENTATION OF DECISION TREE REGRESSOR MODEL FOR PREDICTING THE SALARY OF THE EMPLOYEE

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## EQUIPMENT'S REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM:
1. Import required packages and read the data file.
2. Use LabelEncoder to convert categorical data into numerical data.
3. Split data into testing data and training data.
4. Apply Decision Tree Regressor.
5. Calculate mean squared error and R2.

## PROGRAM:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SHAVEDHA.Y
RegisterNumber:  212221230095
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['Position']=le.fit_transform(data['Position'])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## OUTPUT:
* data.head()  
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427256/fb3764f5-f568-438d-b639-735e8c1339fa)

* data.info()  
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427256/109e631a-4ca3-4f18-85a4-2ebb2e1b0938)

* isnull() and sum()  
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427256/761a0566-651b-4ffe-b4ae-a0a0ed8001a3)

* data.head() for salary  
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427256/f912ebe5-1b88-442e-81cb-a2f5ac290c3d)
 
* MSE value  
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427256/26c719a2-9424-478e-b158-eba104f638a8)

* r2 value  
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427256/564f1170-ebb9-480c-a79d-3f4dfef0d31f)

* data prediction  
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427256/a9f94cc2-4eaf-4265-9377-18c374426ac5)



## RESULT:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
