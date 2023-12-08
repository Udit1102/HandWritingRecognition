This project aims to develop a Machine Learning model using KMeans that recognizes hand written digits.

I have used the data of hand written digits available on UCIML. In this dataset, digits images is present in the form of pixels as features. Each element represents the color value of the pixel.

# Working

1. Loading the data from UCIMLREPO and exploring the same.
2. Generating images from the data for better understanding.
3. Finding the optimal value of clusters by using elbow method and silhouette score. However it is very obvious that we have 10 digits from 0 to 9 so we need ten clusters.
4. Fitting the model with the data and generating cluster centers.
5. Visualising the cluster centers and noting the index position of each cluster.
6. Predicting the sample data with the model. This sample data is array representation of hand written "0553".
