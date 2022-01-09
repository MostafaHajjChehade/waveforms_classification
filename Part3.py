import random
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

dataset_original = pd.read_csv(r'waveform.data', header=None)

df = pd.DataFrame(dataset_original)

correct_output = df[21].values

dataset = df.drop(columns=21).values

pca = PCA(n_components=2)
components = pca.fit_transform(dataset)
fig = px.scatter(components, x=0, y=1, color=correct_output)
fig.show()


while True:
    S = []
    for i in range(0, len(dataset)):
        S.append(i)
    random.shuffle(S)
    S1 = []
    S2 = []
    S1_classes = []
    S2_classes = []
    for j in range(0, int(len(dataset)/2)):
        S1.append(dataset[S[j]])
        S2.append(dataset[S[int(len(dataset)/2) + j]])
        S1_classes.append(correct_output[S[j]])
        S2_classes.append(correct_output[S[int(len(dataset)/2) + j]])

    knn_classifier = KNeighborsClassifier(1)
    knn_classifier.fit(S1, S1_classes)

    predicted2 = knn_classifier.predict(S2)

    new_S2 = []
    new_S2_classes = []
    for i in range(0, int(len(dataset)/2)):
        if predicted2[i] == S2_classes[i]:
            new_S2.append(S2[i])
            new_S2_classes.append(S2_classes[i])

    knn_classifier2 = KNeighborsClassifier(1)
    knn_classifier2.fit(S2, S2_classes)

    predicted1 = knn_classifier2.predict(S1)

    new_S1 = []
    new_S1_classes = []
    for i in range(0, int(len(dataset)/2)):
        if predicted1[i] == S1_classes[i]:
            new_S1.append(S1[i])
            new_S1_classes.append(S1_classes[i])

    new_dataset = []
    new_dataset_classes = []

    for i in range(0, len(new_S2)):
        new_dataset.append(new_S2[i])
        new_dataset_classes.append(new_S2_classes[i])
    for i in range(0, len(new_S1)):
        new_dataset.append(new_S1[i])
        new_dataset_classes.append(new_S1_classes[i])

    dataset = new_dataset
    correct_output = new_dataset_classes

    if new_S1 == S1 and new_S2 == S2:
        break

pca = PCA(n_components=2)
components = pca.fit_transform(dataset)
fig = px.scatter(components, x=0, y=1, color=correct_output)
fig.show()


new_dataset2 = np.c_[dataset, correct_output]


# All the above are the first part of complexity reduction
'''All the above are the first part of complexity reduction'''

storage = []
storage_class = []

nb = int(len(dataset)*random.random())
while correct_output[nb] != 0:
    nb = int(len(dataset) * random.random())
storage.append(dataset[nb])
storage_class.append(correct_output[nb])

nb = int(len(dataset)*random.random())
while correct_output[nb] != 1:
    nb = int(len(dataset) * random.random())
storage.append(dataset[nb])
storage_class.append(correct_output[nb])

nb = int(len(dataset)*random.random())
while correct_output[nb] != 2:
    nb = int(len(dataset) * random.random())
storage.append(dataset[nb])
storage_class.append(correct_output[nb])
a = len(storage)

knn_classifier3 = KNeighborsClassifier(1)
for i in range(0, len(correct_output)):
    #if predicted[i] != new_dataset_classes[i]:
    knn_classifier3.fit(storage, storage_class)
    predicted = knn_classifier3.predict(dataset)
    if predicted[i] != correct_output[i]:
        storage.append(dataset[i])
        storage_class.append(correct_output[i])

print('Size of the final dataset: ' + str(len(storage)))

pca = PCA(n_components=2)
components = pca.fit_transform(storage)
fig = px.scatter(components, x=0, y=1, color=storage_class)
fig.show()
