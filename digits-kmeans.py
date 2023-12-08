import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#loading and exploring the data
digits = load_digits()
#print(digits.feature_names)
#print(digits.data[:4])
#print(digits.DESCR)
#print(digits.target[:3])
#print(digits.target_names)
#exploring the data in the form of image

plt.gray()
#plt.matshow(digits.images[100])
#plt.show()
#print(digits.target[100])

#selecting the optimal num of clusters i.e. 10(0 to 9 digits), just for verification using elbow method and silhouette score

'''k = range(2,14)
inertias = []
sil = []
for i in k:
	model = KMeans(n_clusters=i)
	labels = model.fit_predict(digits.data)
	inertias.append([i, model.inertia_])
	sil.append([i, silhouette_score(digits.data, labels)])
#print(inertias)
#print(sil)'''

#generating clusters with the given data

model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

fig = plt.figure(figsize=(8,3))
fig.suptitle("Cluster Center", fontsize =14, fontweight="bold")
for i in range(10):
    f =fig.add_subplot(5,5, 1+i)
    f.imshow(model.cluster_centers_[i].reshape((8,8)), cmap = plt.cm.binary)
#plt.show()

#predicting the data with sample data i.e. hand written 0553 in array form and labelling the data with names

sample = np.array([
[0.08,3.43,6.56,7.24,7.62,6.94,4.88,0.53,3.66,7.62,5.87,3.81,3.59,4.88,7.55,7.09,4.58,6.94,0.00,0.00,0.00,0.00,1.75,6.94,4.58,6.87,0.00,0.00,0.00,0.00,0.00,3.97,4.57,7.09,0.00,0.00,0.00,0.00,0.00,3.81,3.43,7.63,1.68,0.00,0.00,0.00,0.00,4.04,0.84,7.32,6.10,1.22,0.00,0.61,2.21,6.63,0.00,2.68,7.40,7.62,7.01,7.63,7.62,7.25],
[0.00,0.00,0.00,1.61,4.73,6.10,6.10,4.88,2.67,6.79,7.01,7.62,7.55,5.03,5.03,4.88,4.34,7.62,4.88,2.98,0.69,0.00,0.00,0.00,1.37,7.62,4.27,1.99,3.13,4.43,6.86,7.62,0.00,6.94,7.62,7.62,7.62,7.24,5.57,7.39,0.46,1.37,2.29,2.06,0.92,0.00,2.37,7.62,7.24,4.27,3.81,3.81,3.81,4.73,7.40,6.33,5.41,6.86,6.86,6.86,6.86,6.79,4.96,0.76],
[0.00,0.00,4.19,7.62,7.62,7.62,6.87,0.00,0.00,1.07,6.48,6.48,3.05,3.05,2.29,0.00,0.00,4.50,7.62,2.90,0.00,0.00,0.00,0.00,0.00,2.36,7.62,6.79,7.47,7.62,7.62,7.02,0.00,0.38,6.40,5.65,3.58,3.05,3.43,7.47,0.76,3.74,0.84,0.00,0.00,0.84,5.88,7.62,1.98,7.47,7.32,6.86,7.63,7.62,7.40,2.75,0.00,1.52,3.81,3.81,3.51,3.05,0.99,0.00],
[0.00,0.00,1.67,6.33,6.86,6.86,6.86,5.64,0.00,0.00,1.07,4.58,3.96,3.81,4.96,7.62,0.00,0.00,0.00,0.00,0.00,0.00,2.44,7.62,0.00,0.00,0.15,2.29,4.73,6.56,7.17,7.63,0.00,0.00,5.33,7.62,7.62,7.62,7.62,7.62,0.00,0.00,1.75,3.81,3.81,3.43,7.55,6.10,5.95,3.36,1.52,1.68,3.20,6.48,7.40,1.53,6.25,7.62,7.62,7.62,7.62,6.48,2.06,0.00]
])#0553 in array form

labels = model.predict(sample)

for i in range(len(labels)):
    if labels[i] == 0:
        print(2)
    elif labels[i]== 1:
        print(0) 
    elif labels[i]== 2:
        print(1) 
    elif labels[i]== 3:
        print(8) 
    elif labels[i]== 4:
        print(7) 
    elif labels[i]== 5:
        print(6) 
    elif labels[i]== 6:
        print(3) 
    elif labels[i]== 7:
        print(5) 
    elif labels[i]== 8:
        print(9) 
    elif labels[i]== 9:
        print(4) 


