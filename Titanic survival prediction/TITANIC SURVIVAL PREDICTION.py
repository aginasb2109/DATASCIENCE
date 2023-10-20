import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix

titanic_ds=pd.read_csv("tested.csv")
print(titanic_ds)
titanic_ds.columns

titanic_ds.info()

#Checking null values 
titanic_ds.isnull().sum()
titanic_ds['Age'].describe()
titanic_ds['Age'].fillna(titanic_ds['Age'].median(), inplace=True)

titanic_ds['Fare'].describe()
titanic_ds['Fare'].fillna(titanic_ds['Fare'].median(), inplace=True)

#Dropping Name column , Cabin column and Ticket column as they do not need for survival prediction 
titanic_ds.drop(['Name'], axis=1, inplace=True)
titanic_ds.drop(['Cabin'], axis=1, inplace=True)
titanic_ds.drop(['Ticket'], axis=1, inplace=True)



titanic_ds.isnull().sum()
#converting sex and embarked column values to numbers 
encoder=LabelEncoder()
titanic_ds['Sex']=encoder.fit_transform(titanic_ds['Sex'])
titanic_ds['Embarked']=encoder.fit_transform(titanic_ds['Embarked'])

print(titanic_ds['Sex'])
print(titanic_ds['Embarked'])

#Feature Engineering
titanic_ds['Family']=titanic_ds['SibSp']+titanic_ds['Parch']
titanic_ds['Alone']=(titanic_ds['Family']==0).astype(int)
print(titanic_ds['Family'])
print(titanic_ds['Alone'])


#plots
survived_labels = ["Not Survived", "Survived"]
sns.countplot(x='Sex', hue='Survived', data=titanic_ds, hue_order=[0,1])
plt.title('Survival count by Gender')
plt.xticks([0,1],["Male", "Female"])
plt.legend(title="Survived", labels=survived_labels)
plt.show()


survived_labels = ["Not Survived", "Survived"]
sns.countplot(x='Pclass', hue='Survived', data=titanic_ds, hue_order=[0,1])
plt.title('Survival count by Class')
plt.show()


sns.kdeplot(titanic_ds[titanic_ds['Survived']==0]['Age'], label='Not survived')
sns.kdeplot(titanic_ds[titanic_ds['Survived']==1]['Age'], label='survived')
plt.title('Age distribution by survival')
plt.legend()
plt.show()

#train and test dataset
X=titanic_ds.drop(['Survived'], axis=1)
y=titanic_ds['Survived']
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

#Using Randomforest model
model=RandomForestClassifier()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
accuracy= accuracy_score(y_test, y_pred)
print('Accuracy'+str(accuracy))
confusionmatrix=confusion_matrix(y_test, y_pred)
sns.heatmap(confusionmatrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of Randomforest')
plt.show()


xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, y_train)
y1_pred=xgb_model.predict(X_test)
accuracy1= accuracy_score(y_test, y1_pred)
print('Accuracy'+str(accuracy1))
confusionmatrix1=confusion_matrix(y_test, y1_pred)
sns.heatmap(confusionmatrix1, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of Xgboost')
plt.show()