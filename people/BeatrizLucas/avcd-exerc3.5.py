'''Take 100 samples of 2000 observations with replacement, compute the mean for each sample and plot
   the resulting histogram of means. Test if these 100 mean values are drawn from a normal distribution.'''

import numpy as np
import matplotlib.pyplot as plt

# Generate 100 samples of 2000 observations with replacement
samples = np.random.choice(a=population, size=(100, 2000), replace=True)

# Compute the mean for each sample
sample_means = np.mean(samples, axis=1)

# Plot the histogram of sample means
plt.hist(sample_means, bins=10)
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.title('Histogram of Sample Means')
plt.show()

from scipy.stats import shapiro

# Perform Shapiro-Wilk test
stat, p = shapiro(sample_means)

# Interpret test result
alpha = 0.05
if p > alpha:
    print('Sample means are drawn from a normal distribution')
else:
    print('Sample means are not drawn from a normal distribution')

