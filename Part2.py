import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import time

startTime = time.time()
# read csv file
dataset_original = pd.read_csv(r'waveform.data', header=None)

df = pd.DataFrame(dataset_original)

correct_output = df[21].values

dataset = df.drop(columns=21)

X_to_train = dataset[0:4000]
#print(X_to_train)

X_to_predict = dataset[4000:5000]
#print(X_to_predict)

true_risk = np.zeros(4000)
empirical_risk = np.zeros(4000)

for k in range(1, 4000):
    knn_classifier = KNeighborsClassifier(k)
    knn_classifier.fit(X_to_train, correct_output[0:4000])

    predicted1 = knn_classifier.predict(X_to_train)
    summation1 = 0

    for i in range(0, 4000):
        # print(correct_output[4000+i], predicted[i])
        if correct_output[i] != predicted1[i]:
            summation1 += 1

    empirical_risk[k - 1] = summation1 / 4000

    predicted2 = knn_classifier.predict(X_to_predict)
    summation = 0

    for i in range(0, 1000):
        #print(correct_output[4000+i], predicted[i])
        if correct_output[4000+i] != predicted2[i]:
            summation += 1

    true_risk[k-1] = summation/1000

bla2 = np.zeros(4000)
for j in range(0, 4000):
    bla2[j] = j+1

bayes = np.zeros(4000)
for j in range(0, 4000):
    bayes[j] = 0.14

print(time.time() - startTime)

plt.plot(bla2, true_risk[:], label='True Risk')
plt.plot(bla2, empirical_risk[:], label='Empirical Risk')
plt.plot(bla2, bayes, label='Bayes Error')
plt.xlabel('K')
plt.show()
