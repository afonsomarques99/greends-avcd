import pandas as pd
import numpy as np
import zipfile
import seaborn as sns #for plotting
import matplotlib.pyplot as plt #for plots showing
from statsmodels.graphics.gofplots import qqplot
import random 
from scipy import stats
from scipy.stats import shapiro
from statsmodels.stats import stattools

# Exercise 4.1

''' Take 1000 random samples with replacement of increasing sample sizes (e.g. 10, 50, 100, 150, 200, 250, 
    300, 500 and 1000 observations), compute the mean Temp_ann of each sample and use an appropriate 
    visualization to show how many samples will we need to have a good estimate of the population mean.'''

df = pd.read_csv('people\BeatrizLucas\EFIplus_medit.zip',compression='zip', sep=";")
#print(list(df.columns))

sample_sizes = [10, 50, 100, 150, 200, 250, 300, 500, 1000]
samples_number = 1000
sample_means = []

for size in sample_sizes:
    means = []
    for _ in range(samples_number):  # Perform 1000 iterations of sampling with replacement
        sample_indices = random.choices(range(len(df)), k=size)
        sample = df.iloc[sample_indices]
        mean_temp_ann = sample['temp_ann'].mean()
        means.append(mean_temp_ann)
    sample_means.append(means)

# Calculate mean and standard deviation for each sample size
means = np.mean(sample_means, axis=1)
stds = np.std(sample_means, axis=1, ddof=1)

# Plot 
fig, ax = plt.subplots()
for i in range(len(sample_sizes)):
    ax.scatter(np.repeat(sample_sizes[i], samples_number), sample_means[i], alpha=0.2)
ax.set_xlabel('Sample size')
ax.set_ylabel('Sample mean')
ax.axhline(df['temp_ann'].mean(), color='r', linestyle='--', label='Population mean')
ax.legend()
#plt.show()

# Exercise 4.2.

plot_data = df[['temp_ann', 'Salmo trutta fario']]
plot = sns.catplot(data=plot_data, x='temp_ann', y='Salmo trutta fario', hue='Salmo trutta fario', orient='h', jitter=.25, alpha=.5, legend=False)
plot.set(title='Effect of Mean Annual Temperature on the presence of Salmo trutta fario', xlabel='Annual Temperature (°C)')
plt.legend(['not present', 'present'], loc='center right')
#plt.show()

#Exercise 4.3.

actual_river_slope = df['Actual_river_slope']

# Visualization: Histogram and Q-Q plot

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(actual_river_slope, bins=30, edgecolor='black')
plt.xlabel('Actual River Slope')
plt.ylabel('Frequency')
plt.title('Histogram')

plt.subplot(1, 2, 2)
stats.probplot(actual_river_slope, dist='norm', plot=plt)
plt.title('Q-Q Plot')

plt.tight_layout()
#plt.show()

# Hypothesis Testing: Shapiro-Wilk test

df_2 = df.dropna() 

stat, p = shapiro(pd.Series(df_2['Actual_river_slope']))
print('Statistics=%.3f, p=%.3f' % (stat, p)) 
alpha = 0.05
if p > alpha:
 print('The null hypothesis (data is drawn from a normal distribution) cannot be rejected.')
else:
 print('The null hypothesis (data is drawn from a normal distribution) is rejected.')


# Exercise 4.4

means = []

for i in range(100):
    sample = df_2.sample(2000, replace = True)
    means.append(sample['Actual_river_slope'].mean())

# Plot the histogram of means

sns.histplot(means)
plt.show()

# Perform normality test on the sample means

test_statistic, p_value = shapiro(means)
alpha = 0.05  # Significance level

print(f'Shapiro-Wilk test statistic: {test_statistic}')
print(f'p-value: {p_value}')

if p_value > alpha:
    print('The sample means are drawn from a normal distribution.')
else:
    print('The sample means are not drawn from a normal distribution.')