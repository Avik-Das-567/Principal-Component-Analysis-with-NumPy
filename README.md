# Principal Component Analysis with NumPy

Implementation of **Principal Component Analysis (PCA)** from first principles using **NumPy**. This project demonstrates dimensionality reduction by computing the **covariance matrix**, **eigenvalues**, and **eigenvectors**, and projecting high-dimensional data onto a lower-dimensional subspace.

The workflow is implemented in a **Jupyter Notebook** and applied to the **Iris dataset** to visualize how PCA captures the dominant variance structure of the data.

## Overview

Principal Component Analysis is a **linear dimensionality reduction technique** that transforms a dataset into a new coordinate system where:
- Each axis represents a **principal component**
- Components are **orthogonal**
- Components are ordered by **variance explained**

The primary objective is to represent data in fewer dimensions while preserving as much variance as possible.

This project walks through the **complete PCA pipeline** using NumPy, including data preprocessing, eigen decomposition, explained variance analysis, and projection into a lower-dimensional space.

## Dataset

The project uses the **Iris dataset** from the UCI Machine Learning Repository.

**Features:** Sepal Length, Sepal Width, Petal Length, Petal Width

**Target Classes:** Iris-setosa, Iris-versicolor, Iris-virginica

**Dataset Source:** https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

## PCA Mathematical Foundation

Given a dataset

$X \in \mathbb{R}^{n \times d}$

where 𝑛 = number of samples, and 𝑑 = number of features.

### 1. Standardization
Each feature is centered and scaled:

$X_{\text{scaled}} = \frac{X - \mu}{\sigma}$

This prevents features with larger magnitudes from dominating the variance.

### 2. Covariance Matrix
The covariance matrix describes relationships between features:

$\Sigma = \frac{1}{n - 1} X^{T} X$

It captures how strongly variables vary together.

### 3. Eigen Decomposition
Principal components correspond to the eigenvectors of the covariance matrix:

$\Sigma v = \lambda v$

Where:
- 𝑣 = eigenvector (principal component direction)
- 𝜆 = eigenvalue (variance along that component)

Eigenvalues determine the **importance of each principal component**.

### 4. Singular Value Decomposition
An alternative formulation uses SVD:

$X = U S V^{T}$

The right singular vectors correspond to the **principal component directions**.

### 5. Explained Variance
The proportion of variance explained by each component is:

$\text{Explained Variance Ratio} = \frac{\lambda_i}{\sum_j \lambda_j}$

The cumulative sum helps determine how many components should be retained.

### 6. Projection
Data is projected onto the selected principal components:

$X_{PCA} = XW$

where 𝑊 is the projection matrix composed of the selected eigenvectors.

## Implementation Workflow
The notebook follows a structured pipeline.

### 1. Load Data and Libraries
Libraries used include:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
The Iris dataset is loaded directly from the UCI repository.

### 2. Exploratory Visualization
A scatter plot is used to inspect the structure of the dataset.
```
sns.scatterplot(
    x=iris.sepal_length,
    y=iris.sepal_width,
    hue=iris.species,
    style=iris.species
)
```
This helps visually assess potential class separability.

### 3. Data Standardization
Features are standardized using:
```
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
```
This ensures each feature contributes equally to the covariance calculation.

### 4. Covariance Matrix Computation
```
covariance_matrix = np.cov(X.T)
```
The covariance matrix represents pairwise feature relationships.

### 5. Eigen Decomposition
Eigenvalues and eigenvectors are computed using NumPy:
```
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
```
- Eigenvectors represent principal directions
- Eigenvalues quantify variance captured by each direction

### 6. Singular Value Decomposition
SVD provides an alternative computation:
```
eigen_vec_svd, s, v = np.linalg.svd(X.T)
```
This approach is often more numerically stable for large datasets.

### 7. Explained Variance Analysis
Variance contribution for each component:
```
variance_explained = [(i / sum(eigen_values)) * 100 for i in eigen_values]
```
Cumulative variance:
```
cumulative_variance_explained = np.cumsum(variance_explained)
```
A plot is generated to determine how many components preserve sufficient variance.

### 8. Projection to Lower Dimensions

The projection matrix is constructed using the top eigenvectors:
```
projection_matrix = (eigen_vectors.T[:][:])[:2].T
```
The dataset is transformed into PCA space:
```
X_pca = X.dot(projection_matrix)
```

### 9. PCA Visualization
The transformed data is visualized in 2D:
```
sns.scatterplot(
    x=X_pca[y==species, 0],
    y=X_pca[y==species, 1]
)
```
This reveals how PCA separates the classes in a reduced dimensional space.

## Results
Key observations from the PCA projection:
- The **first principal component captures the largest variance**.
- The **first two components contain the majority of dataset variance**.
- PCA enables clear visualization of class clusters in **two dimensions**.

The Iris species show significant separation after projection, demonstrating the effectiveness of PCA for dimensionality reduction.

## Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn (StandardScaler)

## Key Concepts Demonstrated
- Feature standardization
- Covariance matrix computation
- Eigen decomposition
- Singular Value Decomposition
- Explained variance analysis
- Linear subspace projection
- PCA visualization

---
