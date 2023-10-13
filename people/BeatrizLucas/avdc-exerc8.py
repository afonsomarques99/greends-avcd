import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

# Exercise 8.1

# Database 

df = pd.read_csv('people\BeatrizLucas\EFIplus_medit.zip',compression='zip', sep=";")
df = df.dropna()

# Subset the database - Douro and Tejo basins

df = df[ (df['Catchment_name'] == 'Tejo') | (df['Catchment_name'] == "Douro")]

# Subset the database - environmental variables

env_var = ['Altitude', 'Actual_river_slope', 'Elevation_mean_catch', 'prec_ann_catch', 'temp_ann', 'temp_jan', 'temp_jul']
df = df[env_var]
df = df.reset_index(drop=True)
print(df)

# Standardize the data

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Perform PCA

pca = PCA()
pca.fit(df_scaled)

# Get the principal components and explained variance ratio

principal_components = pca.transform(df_scaled)
explained_variance_ratio = pca.explained_variance_ratio_

# Print the principal components and explained variance ratio
print("Principal Components:")
print(principal_components)
print()
print("Explained Variance Ratio:")
print(explained_variance_ratio)

#Exercise 8.2

# Database 

df = pd.read_csv('people\BeatrizLucas\EFIplus_medit.zip',compression='zip', sep=";")
df = df.dropna()

# Subset the database - Douro and Tejo basins

df = df[ (df['Catchment_name'] == 'Tejo') | (df['Catchment_name'] == "Douro")]

# Select only numeric columns
numeric_columns = df.select_dtypes(include=np.number)

# Calculate dissimilarity matrix
dissimilarity_matrix = pairwise_distances(numeric_columns, metric='euclidean')

# Perform MDS
mds = MDS(n_components=2, dissimilarity='precomputed')
mds_result = mds.fit_transform(dissimilarity_matrix)

# Perform NMDS
nmds_result = MDS(n_components=2, dissimilarity='precomputed', metric=False, n_init=100).fit_transform(dissimilarity_matrix)

# Normalize the coordinates for better visualization
scaler = MinMaxScaler()
mds_result_scaled = scaler.fit_transform(mds_result)
nmds_result_scaled = scaler.fit_transform(nmds_result)

# Plotting MDS results
sns.scatterplot(x=mds_result_scaled[:,0],
              y=mds_result_scaled[:,1],
              hue = df['Catchment_name'].tolist(),
              linewidth=0,
              )



# Plotting NMDS results
sns.scatterplot(x=nmds_result_scaled[:,0],
              y=nmds_result_scaled[:,1],
              hue = df['Catchment_name'].tolist(),
              linewidth=0,
              )

plt.show()