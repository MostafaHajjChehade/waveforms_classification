# importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# reading dataset
data = np.genfromtxt("Data.txt", delimiter=",", usemask=True)

# number of samples in our dataset is 5000
# number of features are 40
# data array shape is (5000, 41)
# Y labels were included in the last column

# separating X and Y
Data_X, Data_Y = data[:,:data.shape[1] - 1], data[:,data.shape[1] - 1]

# Data_X.shape (5000, 40)
# Data_Y.shape (5000,)



# Separating our dataset into train_set and test_set 
X_train, X_test, y_train ,y_test = train_test_split(Data_X, Data_Y, test_size=0.2)

# X_train.shape  (4000, 40)
# y_train.shape  (4000,)
# X_test.shape   (1000, 40)
# y_test.shape   (1000,)


max_k = 800
folds = 10
best_k = None
max_acc = 0
num_neighbours = np.arange(1, max_k + 1)
all_train_accs = []

for k in num_neighbours:

    kNN = KNeighborsClassifier(n_neighbors=k)
    accs = cross_val_score(kNN, X_train, y_train, cv=folds, scoring= 'accuracy')
    avg_acc = np.mean(accs)
    print("K is ", k, "Val-Accuracy is ", avg_acc)

    all_train_accs.append(avg_acc)
    
    if avg_acc > max_acc:
        best_k = k
        max_acc = avg_acc
        print("best K changed to ", k, "Max val-accuracy is ", avg_acc)

kNN_best = KNeighborsClassifier(n_neighbors= best_k)
kNN_best.fit(X_train, y_train)
y_preds = kNN_best.predict(X_test)
test_acc = accuracy_score(y_test, y_preds)

print("\n\nBest k is ", best_k)
print("Test accuracy is ", test_acc)

xt = [i for i in range(0, max_k + 1, max_k // 4)]
xt.append(best_k)

plt.title('kNN_Accuracy over Number of Neighbours')
plt.plot(num_neighbours, all_train_accs, label = 'Val-Acc')
plt.axvline(x= best_k, label = 'best K', c='r', linestyle='--')
plt.axhline(y= test_acc, color='orange', label = 'Test-Acc', linestyle='-.')
plt.axhline(y= 0.86, color='g', label = 'Optimal-Bayes-Acc', linestyle=':')
plt.legend()
plt.xlabel('Number of Neighbours')
plt.ylabel('Accuracy')
plt.xticks(sorted(xt))

plt.show()