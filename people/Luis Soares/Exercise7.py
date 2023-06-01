import pandas as pd
import os
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt

# Exercise 7.1

df = pd.read_csv('people\BeatrizLucas\EFIplus_medit.zip',compression='zip', sep=";")
df = df.dropna()

# Subset the database - Douro and Tejo basins

df = df[ (df['Catchment_name'] == 'Tejo') | (df['Catchment_name'] == "Douro")]

# Subset the database - environmental variables

env_var = ['Altitude', 'Actual_river_slope', 'Elevation_mean_catch', 'prec_ann_catch', 'temp_ann', 'temp_jan', 'temp_jul']
df = df[env_var]
df = df.reset_index(drop=True)
print(df)

# Agglomerative clustering using different linkage methods
linkage_methods = ['ward', 'complete', 'average', 'single']  # List of linkage methods to try

for method in linkage_methods:
    
    clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage=method)

    labels = clustering.fit_predict(df)

    # Print the cluster labels
    print(f"Cluster labels using {method} linkage:")
    print(labels)
    print()

# Exercise 7.2. 

sns.clustermap(df, col_cluster=False, row_cluster=True, method='average')
plt.show()