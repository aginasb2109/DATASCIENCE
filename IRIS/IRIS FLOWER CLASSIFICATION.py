import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


iris_data=pd.read_csv("IRIS.csv")
print(iris_data.head(6))
iris_data.columns

iris_data.isnull().sum()
species = ["setosa", "versicolor", "virginica"]
species_counts = iris_data['species'].value_counts().sort_index()
plt.bar(species, species_counts, color=['blue', 'green', 'red'])
print(species_counts)

X=iris_data.drop('species', axis=1)
y=iris_data['species']

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=42)

logistic=LogisticRegression(max_iter=1000)
logistic.fit(X_train, y_train)

rf=RandomForestClassifier()
rf.fit(X_train, y_train)

decisiontree=DecisionTreeClassifier()
decisiontree.fit(X_train, y_train)

svc= SVC()
svc.fit(X_train, y_train)

y_preditlog=logistic.predict(X_test)
y_preditrf=rf.predict(X_test)
y_preditdt=decisiontree.predict(X_test)
y_preditsvc=svc.predict(X_test)

accuracy_log= accuracy_score(y_test, y_preditlog )
report_log= classification_report(y_test, y_preditlog )
print(accuracy_log)
print(report_log)


accuracy_rf= accuracy_score(y_test, y_preditrf )
report_rf= classification_report(y_test, y_preditrf )
print(accuracy_rf)
print(report_rf)

accuracy_dt= accuracy_score(y_test, y_preditdt )
report_dt= classification_report(y_test, y_preditdt )
print(accuracy_dt)
print(report_dt)

accuracy_svc= accuracy_score(y_test, y_preditsvc )
report_svc= classification_report(y_test, y_preditsvc )
print(accuracy_svc)
print(report_svc)
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVC']
accuracies = [accuracy_log, accuracy_dt, accuracy_rf, accuracy_svc]

correlation_matrix = iris_data.iloc[:, :-1].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

scatter_data = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
sns.set(style="ticks")
sns.pairplot(scatter_data, markers="o")
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.title('Model Accuracies')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0.9, 1.0) 
plt.show()






