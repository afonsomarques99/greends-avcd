import pandas as pd
import scipy.stats as sts
import seaborn as sns 
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency

# Exercise 5.1

df = pd.read_csv('people\BeatrizLucas\EFIplus_medit.zip',compression='zip', sep=";")

# Contingency table
contingency_table = pd.crosstab(df['Salmo trutta fario'], df['Country'])

# Chi-square test of independence
chi2, p_value, _, _ = chi2_contingency(contingency_table)

# Significance level
alpha = 0.05

# Hyphotesis test
if p_value < alpha:
    print("The frequency of sites with presence and absence of Salmo trutta fario is dependent on the country.")
    print("Null hypothesis rejected.")
else:
    print("The frequency of sites with presence and absence of Salmo trutta fario is independent of the country.")
    print("Null hypothesis cannot be rejected.")


# Exercise 5.2

df_2 = df.dropna()

data = df_2[['Salmo trutta fario','Actual_river_slope']]

# Subset the data 

presence_data = df_2[df_2['Salmo trutta fario']==1]['Actual_river_slope']
absence_data = df_2[df_2['Salmo trutta fario']==0]['Actual_river_slope']

#Means/medians

mean_presence = presence_data.mean()
mean_absence = absence_data.mean()

# Perform mann whitney test

stat, p = sts.mannwhitneyu(mean_absence, mean_absence, alternative='two-sided')

print('stat=%.3f, p-value=%.3f' % (stat, p))

alpha = 0.05
if p <= alpha:
 print('Null hypothesis reject')
 print("The means of “Actual_river_slope” between presence and absence sites of Salmo trutta fario are not equal")
else:
 print('Null hypothesis not reject')
 print("The means of “Actual_river_slope” between presence and absence sites of Salmo trutta fario are equal")

