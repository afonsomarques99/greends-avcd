import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

#Exercise 9.1
#Database
df = pd.read_csv('people\Luis Soares\EFIplus_medit.zip',compression='zip', sep=";")
df = df.dropna()

#Subset the database - Douro and Tejo basins
df_basins = df[ (df['Catchment_name'] == 'Tejo') | (df['Catchment_name'] == "Douro")]

#Subset the database - environmental variables
env_var = ['Altitude', 'Actual_river_slope', 'Elevation_mean_catch', 'prec_ann_catch', 'temp_ann', 'temp_jan', 'temp_jul']
df_env_var = df[env_var]
df = df.reset_index(drop=True)
print(df)

#Separate the features (environmental variables) and the target variable (Catchment_name)
X = df[env_var]
y = df['Catchment_name']

#Encode the target variable labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#Perform Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y_encoded)

#Create a scatter plot of the LDA transformed features with color-coded classes
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_encoded)

#Add arrows indicating the contribution of each feature to the LDA dimensions
featurecontributions = pd.DataFrame(lda.coef.T, columns=['LDA1', 'LDA2'], index=env_var)
for feature in feature_contributions.index:
    plt.arrow(0, 0, feature_contributions.loc[feature, 'LDA1'], feature_contributions.loc[feature, 'LDA2'],
              color='black', head_width=0.05)
    plt.text(feature_contributions.loc[feature, 'LDA1'] * 1.1,
             feature_contributions.loc[feature, 'LDA2'] * 1.1,
             feature, color='black', ha='center', va='center')

#Set plot title and labels
plt.title('Linear Discriminant Analysis Biplot')
plt.xlabel('LDA1')
plt.ylabel('LDA2')

#Show the plot
plt.show()
