import numpy as np
import time
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("waveform.csv")
#data = data.sort_values("label")
#print(data.head())

group_label = data.groupby("label")

#class0 = 50data
grouped_class0 = group_label.get_group(0)
grouped_class0 = grouped_class0.head(50)

#class1 = 50data
grouped_class1 = group_label.get_group(1)
grouped_class1 = grouped_class1.head(50)

#class2 = 50data
grouped_class2 = data.loc[data['label'] == 2]

imbalanced_data = pd.concat([grouped_class0, grouped_class1, grouped_class2])
#print(imbalanced_data)

#Create x and y variables.
X = imbalanced_data.drop(['label'], axis=1).values
y = imbalanced_data['label']
print(y.value_counts())
y.value_counts().plot.pie(autopct='%1.1f%%')
plt.show()

#print("**X**\n",X)
#print("**Y**\n",y)

    #List Hyperparameters that we want to tune.
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn_2 = KNeighborsClassifier()
#Use GridSearch
clf = GridSearchCV(knn_2, hyperparameters, cv=10)
#Fit the model
best_model = clf.fit(X,y)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
print("##################################################")


#Split data into training and testing.
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,train_size=0.8)
#print("**X_train**\n",X_train)

# Fitting K-NN to the Training set

classifier = KNeighborsClassifier(n_neighbors = best_model.best_estimator_.get_params()['n_neighbors'],weights='distance')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#print('y_pred',y_pred)
#print('y_test',y_test)

#MODEL EVALUATION STEP
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n\n########Confusion Matrix########")
print(cm)
print("\n\n########F-Measure########")
print(f1_score(y_test, y_pred, average=None))
print("micro")
print(f1_score(y_test, y_pred, average='micro'))
print("macro")
print(f1_score(y_test, y_pred, average='macro'))
print("weighted")
print(f1_score(y_test, y_pred, average='weighted'))
print("##################################################")


print(metrics.classification_report(y_test, y_pred, digits=3))
sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False)
plt.title('confusion matrix')
plt.xlabel('true class')
plt.ylabel('predicted class')
plt.show()
#print(classifier.classes_)

