import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import  euclidean_distances

#Exercise 9.1

# Database 

df = pd.read_csv('people\BeatrizLucas\EFIplus_medit.zip',compression='zip', sep=";")
df = df.dropna()

# Subset the database - Douro and Tejo basins

df = df[ (df['Catchment_name'] == 'Tejo') | (df['Catchment_name'] == "Douro")]

# Add new columns 

new_columns = ['Catchment_name', 'Altitude', 'Actual_river_slope', 'Elevation_mean_catch', 'prec_ann_catch', 'temp_ann', 'temp_jan', 'temp_jul']
df = df[new_columns]

df = df.dropna()
df = df.reset_index(drop = True)

#Subset
subset = df.drop("Catchment_name", axis=1)

# Standardize
data_std = StandardScaler().fit_transform(subset)

# PCA
pca = PCA(n_components=2)
pca_fit = pca.fit_transform(data_std)
pca_df = pd.DataFrame(data=pca_fit, columns=["PC1", "PC2"])

# Biplot
fig, ax = plt.subplots(figsize=(10, 10))

# Convert Catchment_name to numeric values for color mapping
unique_catchments = df["Catchment_name"].unique()
catchment_mapping = {c: i for i, c in enumerate(unique_catchments)}
colors = [catchment_mapping[c] for c in df["Catchment_name"]]

# Scatter plot of PC1 and PC2
ax.scatter(pca_df["PC1"], pca_df["PC2"], c=colors, cmap="viridis")

variables = pca.components_.T * np.sqrt(pca.explained_variance_)
variable_labels = subset.columns

for i, v in enumerate(variables):
    ax.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1, color="r")
    ax.text(v[0]*1.15, v[1]*1.15, variable_labels[i], color="r")

ax.set_xlabel("PC1 ({:.1f}% explained variance)".format(pca.explained_variance_ratio_[0]*100))
ax.set_ylabel("PC2 ({:.1f}% explained variance)".format(pca.explained_variance_ratio_[1]*100))
ax.set_title("PCA Biplot of Quantitative Environmental Variables in Douro and Tejo Basins")

plt.show()

#Exercise 9.2

columns = ["Altitude", "Actual_river_slope", "Elevation_mean_catch", "prec_ann_catch", "temp_ann", "temp_jan", "temp_jul"]

sns.pairplot(df[columns], hue="Catchment_name")

plt.show()