import pandas as pd
import zipfile
import seaborn as sns # For plotting
import matplotlib.pyplot as plt # For showing plots




df = pd.read_csv('EFIplus_medit.zip',compression='zip', sep=";")

# clean up the dataset to remove unnecessary columns (eg. REG) 
df.drop(df.iloc[:,5:15], axis=1, inplace=True)

# let's rename some columns so that they make sense
df.rename(columns={'Sum of Run1_number_all':'Total_fish_individuals'}, inplace=True) # inplace="True" means that df will be updated

# for sake of consistency, let's also make all column labels of type string
df.columns = list(map(str, df.columns))

# Check data types
pd.options.display.max_rows = 154 # maximum number of rows displayed.
df.dtypes

ctab = pd.crosstab(df['Eutrophication'], df['Salmo trutta fario'])
print(ctab)

? numpy
