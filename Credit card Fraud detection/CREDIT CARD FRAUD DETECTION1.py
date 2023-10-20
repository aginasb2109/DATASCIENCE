import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from imblearn.over_sampling import SMOTE

data= pd.read_csv("creditcard.csv")
print(data)

print(data.head())
print(data.tail(5))

print(data.info())

print(data.isnull().sum())

X = data.drop('Class', axis=1)
y = data['Class']

#Training and Testing the dataset

X_train, X_test, y_train, y_test= train_test_split (X, y, test_size=0.3, random_state=3)
smote = SMOTE(sampling_strategy='auto', random_state=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


#LOGISTIC REGRESSION
logreg = LogisticRegression(max_iter=1000)  
logreg.fit(X_train_resampled, y_train_resampled)
log_pred=logreg.predict(X_test)
print(log_pred)

#RANDOMFOREST REGRESSION
ranreg=RandomForestClassifier()
ranreg.fit(X_train_resampled, y_train_resampled)
ran_pred=ranreg.predict(X_test)
print(ran_pred)

#ACCURACY MEASURE FOR LOGISTIC REGRESSION
log_accuracy=accuracy_score(y_test, log_pred)
log_precision=precision_score(y_test, log_pred)
log_recall=recall_score(y_test, log_pred)
log_f1=f1_score(y_test, log_pred)


#ACCURACY MEASURE FOR RANDOMFOREST REGRESSION
rf_accuracy=accuracy_score(y_test, log_pred)
rf_precision=precision_score(y_test, log_pred)
rf_recall=recall_score(y_test, log_pred)
rf_f1=f1_score(y_test, log_pred)

print("Logistic regression accuracy:{}".format( log_accuracy))
print("Logistic regression precision:{}".format(log_precision))
print("Logistic regression recall:{}".format(log_recall))
print("Logistic regression f1:{}".format( log_f1))

print("Randomforest regression accuracy:{}".format(rf_accuracy))
print("Randomforest regression precision:{}".format(rf_precision))
print("Randomforest regression recall:{}" .format(rf_recall))
print("Rogistic regression f1:{}".format(rf_f1))

#PLOT TO DIFFERENTIATE LEGAL AND FRAUD FOR V1 AND V2
plt.scatter(data[data['Class']==0]['V1'],data[data['Class']==0]['V2'], c='green' , label='Legal' )
plt.scatter(data[data['Class'] == 1]['V1'], data[data['Class'] == 1]['V2'], c='red', label='Fraud')
plt.xlabel("V1")     
plt.ylabel("V2")   
plt.title("Legal and Fraud Transactions plot")  
plt.legend()
plt.show()      

                       


