import pandas as pd
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('people\BeatrizLucas\EFIplus_medit.zip',compression='zip', sep=";")
df = df.dropna()
pd.options.display.max_seq_items = 200
print(df.columns)

#Find where the columns related to the number of individuals for each species start

first_column = df.columns.get_loc("Total_sp")+1
print("Column nº", first_column)
df["sp_rich"]=df[df.iloc[:,first_column:] >= 1].count(axis=1)
print(sp_rich)
columns = ["sp_rich", "Altitude", "Actual_river_slope", "Elevation_mean_catch", "prec_ann_catch", "temp_ann", "temp_jan", "temp_jul"]
df_2 = df[columns]
df_2.head()
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,8))

for i, col in enumerate(columns):
    sns.histplot(df_2[col], ax=axes[i//4,i%4], kde=True)

plt.tight_layout()
plt.show()
df_2['Actual_river_slope'] = np.log10(df_2['Actual_river_slope']+1)
df_2.rename(columns={'Actual_river_slope': 'log10_Actual_river_slope'}, inplace = True)
columns = list(df_2)

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,8))

for i, col in enumerate(columns):
    sns.histplot(df_2[col], ax=axes[i//4,i%4], kde=True)

plt.tight_layout()
plt.show()
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,8))

for i, col in enumerate(columns[1:]):
    
    model = smf.ols(formula = f'sp_rich ~ {col}', data=df_2)
    model = model.fit()

    # get the coefficient estimates, R-squared, F-statistics

    est_coeffs = model.params
    R2 = model.rsquared
    F = model.fvalue
    p = model.f_pvalue

    # Generate the predicted values from the fitted model

    predicted = model.predict(df_2[col])

    # Plot the original data points and the predicted values

    nrow = i // 4
    ncol = i % 4
    ax = axes[nrow][ncol]
    sns.scatterplot(x = df_2[col], y = df_2['sp_rich'], ax = ax)
    ax.plot(df_2[col], predicted, color='red')
    ax.set_xlabel(f'{col} \n\n intercept={est_coeffs[0]:.2f}, slope={est_coeffs[1]:.2f} \n R^2 = {R2:.2f}, F-stat = {F:.0f}, p = {p:.2f}')
    ax.set_ylabel('sp_rich')
    ax.set_title(f'~{col}')

plt.subplots_adjust(hspace = 0.5)
plt.show()
X = df_2[columns[1:]]
y = df_2['sp_rich']

X = sm.add_constant(X) 
model = sm.OLS(y, X).fit()

print(f'R^2 = {model.rsquared:.2f}, F-stat = {model.fvalue:}, p = {model.f_pvalue:.2f}')
print(model.summary())

fig = sm.graphics.plot_partregress_grid(model)

predictor_vars = ['Altitude', 'log10_Actual_river_slope', 'Elevation_mean_catch', 'prec_ann_catch', 'temp_ann', 'temp_jan', 'temp_jul']
X = df_2[predictor_vars]
y = df_2['sp_rich']

vif = pd.DataFrame()
vif['Predictor'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)
X2 = X.drop(['temp_ann', 'temp_jan', 'temp_jul'], axis=1)
model2 = sm.OLS(y, sm.add_constant(X2)).fit()

coeffs = pd.DataFrame()
coeffs['MLR1']= model.params
coeffs['MLR2'] = model2.params
print(coeffs)
X3 = X2.drop(['Elevation_mean_catch', 'prec_ann_catch'], axis=1)
model3 = sm.OLS(y, sm.add_constant(X3)).fit()

coeffs['MLR3'] = model3.params
print(coeffs)