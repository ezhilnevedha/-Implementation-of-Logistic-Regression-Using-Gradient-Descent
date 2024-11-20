# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Dataset  
2. Inspect and Explore the Dataset  
3. Drop Irrelevant Columns  
4. Convert Categorical Variables to Numeric  
5. Encode Categorical Variables with Integer Codes  
6. Prepare Features (`X`) and Target (`Y`) Variables  
7. Define Logistic Regression Functions: Sigmoid, Loss, Gradient Descent  
8. Train the Logistic Regression Model Using Gradient Descent  
9. Predict and Evaluate Model Accuracy  
10. Test Model on New Data Points  

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Ezhil Nevedha.K
RegisterNumber:  212223230055
*/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
dataset=pd.read_csv('Placement.csv')
print(dataset)
dataset.head()
dataset.tail()
dataset.info()
dataset.drop('sl_no',axis=1,inplace=True)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
dataset.info()
dataset.head()
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
X
Y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,X,Y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(x.dot(theta))
    gradient=x.T.dot(h-y)/m
    theta-=alpha*gradient
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01, num_iterations=1000)
def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(Y)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

```

## Output:

![374863252-acdf306f-06cb-4a2b-98ef-ba70a0487cc4](https://github.com/user-attachments/assets/c038fdc6-a8c8-4e1b-8dc8-a0ef5918fcc9)

![374863502-a815db95-643c-4f10-8b0a-3041e484f42d](https://github.com/user-attachments/assets/82cf8757-0a12-4d56-ac95-f6bd655012b6)

![374863697-8e7490bd-0fb2-4368-871c-770c336f98ba](https://github.com/user-attachments/assets/8758a5a2-429f-4f1e-b606-c0e560b2b0ca)

![374864073-19e73724-93b1-4c19-b947-4d19925f42d7](https://github.com/user-attachments/assets/83cff383-f0bc-463e-8248-784c38368d8b)

![374864379-7bd6179a-135f-4626-82fb-1de700f2b8e0](https://github.com/user-attachments/assets/aefaae5c-ad85-4697-90b3-6ccca73ac466)

![374865103-46e63937-2a28-4d14-a341-3c0fbe65f47e](https://github.com/user-attachments/assets/ae1252f6-1dfa-4a84-b064-b935b2ec4d19)

![374865290-dfdbba5c-e993-4c05-a06d-2c7a2140b017](https://github.com/user-attachments/assets/a5487666-268f-4485-be1e-1d9c964538bf)

![374865489-9bc09834-e043-435d-b3a6-2e0d6f71fd50](https://github.com/user-attachments/assets/34e9098b-ed72-4eeb-86d8-041c06b72702)

![374866513-a1776a01-f646-41de-be92-3617743db71c](https://github.com/user-attachments/assets/8db1b3fd-ad17-474d-8ea4-e7b1c646ae00)

![374868527-69f7bb24-d488-4a22-961f-be46e9499808](https://github.com/user-attachments/assets/c17ea379-906c-42bf-95e6-5135daf9840b)

![374869039-d03776ee-dcbf-45e3-be6b-f690ec6944a2](https://github.com/user-attachments/assets/e2c049ec-7b39-4dbe-95d3-13ed223b97c4)

![374869714-9a956ce3-0ac8-432a-9b09-5a18d5d8d37e](https://github.com/user-attachments/assets/046ce566-8df3-4c9d-b180-2bbdca4851b2)
![374870514-620e69af-82b7-4da0-8d54-ccb10843ca15](https://github.com/user-attachments/assets/e09eefd0-86ea-47b0-a97f-34f33fc53961)

![374873483-c96429e4-ab59-41cb-b2d3-4a37f3a2d38e](https://github.com/user-attachments/assets/f8c5b5d9-525d-4f4a-8e99-ab5adbecb8f0)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

