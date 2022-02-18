#!/usr/bin/env python
# coding: utf-8

# # Maria David

# ### Task-2 Prediction Using Unsupervised ML

# ### K-Means Clustering

# ---

# ##### Introduction

# K-Means clustering groups similar data points together and discover underlying patterns. K-Means Clustering is an Unsupervised Learning algorithm, which groups the unlabeled dataset into different clusters. Here K defines the number of pre-defined clusters .K-means is a centroid-based algorithm, or a distance-based algorithm, where we calculate the distances to assign a point to a cluster. In K-Means, each cluster is associated with a centroid.

# ------

# ##### Objective

# To predict the optimum number of clusters and represent it visually using the Iris dataset

# -----

# ##### About the dataset

# The iris dataset contains information about the 4 different features Sepal length ,Sepal width, Petal length and Petal Width of different species. The variables of the dataset are "Id","SepalLengthCm","SepaWidthCm","PetalLengthCm","PetalWidthCm" and "Species" .

# ----

# ##### Step 1:Importing the libraries

# In[14]:


import pandas as pd
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt


# ##### Step 2:Loading the dataset

# In[34]:


iris_data = pd.read_csv('Iris.csv')


# ##### Step 3:Understanding the data

# In[35]:


iris_data.head()


# The dataset consists of 1 dependent variable(Species) and 4 independent variables

# In[36]:


iris_data.shape


# There are 150 observations and 6 variables in the dataset "Iris"

# In[37]:


iris_data.describe()


# The average Sepal length of the flowers is around 5 cm.
# The average Sepal width of the flowers is around  3 cm.
# The average Petal length of the flowers is around 3.7 cm.
# The average Petal width of the flowers is around 1 cm.

# In[38]:


iris_data.info()


# In[51]:


#Frequency distribution of species
No_species = pd.crosstab(index=iris_data["Species"],  # Make a crosstab
                              columns="count")      # Name the count column

No_species


# There are 3 types of species and 50 each.

# ##### Step 4:Data Cleaning

# In[42]:


#a.Removing unwanted columns
iris_data=iris_data.drop(["Id"],axis=1)


# In[43]:


#b.Checking for null values
iris_data.isnull().sum()


# There are no null values in the data.

# In[45]:


#c.Checking for outliers
iris_data.plot.box(title="Boxplot of all columns",figsize=(10,8))


# There are outliers in the sepal width column ,but we do not remove them because there are small in number and do not affect the analysis.

# ##### Step 5:Exploratory data anlaysis

# In[48]:


#Understanding the distribution

plt.rcParams['figure.figsize'] = (15, 15)
plt.subplot(2, 2, 1)
sns.set(style = 'whitegrid')
sns.distplot(iris_data['SepalLengthCm'])
plt.title('Distribution of sepal length (cm)', fontsize = 15)
plt.xlabel('SepalLengthCm')
plt.ylabel('Count')

plt.subplot(2, 2, 2)
sns.set(style = 'whitegrid')
sns.distplot(iris_data['SepalWidthCm'], color = 'red')
plt.title('Distribution of sepal width (cm)', fontsize = 15)
plt.xlabel('SepalWidthCm')
plt.ylabel('Count')

plt.subplot(2, 2, 3)
sns.set(style = 'whitegrid')
sns.distplot(iris_data['PetalLengthCm'], color = 'green')
plt.title('Distribution of petal length (cm)', fontsize = 15)
plt.xlabel('PetalLengthCm')
plt.ylabel('Count')

plt.subplot(2, 2, 4)
sns.set(style = 'whitegrid')
sns.distplot(iris_data['PetalWidthCm'], color = 'yellow')
plt.title('Distribution of petal width (cm)', fontsize = 15)
plt.xlabel('PetalWidthCm')
plt.ylabel('Count')


plt.show()


# Sepal length lies between 4 and 8 cms,Sepal width lies between 2 to 4.5 ,petal length lies between 1 to 7,Petal width lies between 0 and 3 cm. There is a dip in the petal length in the range of 2 to 4 cms. There is a dip in the petal width from 0.5 to 1cms.

# In[56]:


#pairplot
plt.figure(1,figsize=(15,7))
n=0
for x in ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']:
    for y in ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']:
        n+=1
        plt.subplot(4,4,n)
        plt.subplots_adjust(hspace=0.5,wspace=0.5)
        sns.regplot(x=x,y=y,data=iris_data)
        plt.ylabel(y.split()[0]+''+y.split()[1] if len(y.split())>1 else y)
plt.show()


# There is a positive relationship between petal length and sepal length,petal length and petal width.There is a negative relation between petal length and sepal width.

# ##### Step 6:Finding the optimum number of clusters for K Means

# In[57]:


# Finding the optimum number of clusters for k-means classification using the Elbow method
#The elbow method allows us to pick the optimum no. of clusters for classification.
plt.figure(1,figsize=(15,7))
x = iris_data.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# From the above graph we can see that the optimal no.of clusters is 3.Therefore, k=3.

# ##### Step 7:KMeans Clustering

# We select k random points from the data as centroids ,Assign all the points to the closest cluster centroid
# and recompute the centroids of newly formed clusters and repeat the steps.

# In[52]:


# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[53]:


# Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# There are 3 clusters in blue,red and green and yellow points are the centroids of each groups.

# ----

# ##### Conclusion

# The above analysis depicts k-means clustering by grouping the data points into 3 different clusters.

# In[ ]:


THANK YOU!!

