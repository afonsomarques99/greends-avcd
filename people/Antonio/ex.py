# Analysis and Visualization of Complex Data
## Anscombe's quartet

"""The `Anscombe's quartet` is a classical example of how important is data visualization when analysing data. 
It is often used to illustrate the importance of looking at a set of data graphically and not only relying on basic statistic properties.
1. Run this code first
```
%matplotlib inline # means that for inline plotting, commands in cells below the cell that outputs a plot will not affect the plot.
```
2. Produce datasets

"""
import matplotlib.pyplot as plt
import numpy as np

# Produce lists
x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

# Produce dictionary
datasets = {
    'I': (x, y1),
    'II': (x, y2),
    'III': (x, y3),
    'IV': (x4, y4)
}

#3. Plot scatterplots for each dataset, showing the regression lines, the mean, standard deviation and correlation coefficient (r)


fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6), # number of graphs per column and line, share axis or not and figure size
                        gridspec_kw={'wspace': 0.08, 'hspace': 0.08}) # vertical and horizontal spaces between graph cells, respectively
axs[0, 0].set(xlim=(0, 20), ylim=(2, 14)) # limits for x-axis and y-axis.
axs[0, 0].set(xticks=(0, 10, 20), yticks=(4, 8, 12)) # tickmarks for x-axis and y-axis.

for ax, (label, (x, y)) in zip(axs.flat, datasets.items()):
    ax.text(0.1, 0.9, label, fontsize=20, transform=ax.transAxes, va='top') # Add dataset name on the top left side
    ax.tick_params(direction='in', top=True, right=True) # add tickmarks including on top and right sides
    ax.plot(x, y, 'o') # ass data points ('o' means circles)
        # linear regression
    p1, p0 = np.polyfit(x, y, deg=1)  # estimate intercept (p0) and slope (p1) 
    ax.axline(xy1=(0, p0), slope=p1, color='r', lw=2) # draw regression line (red)

    # add text box for the statistics
    stats = (f'$\\mu$ = {np.mean(y):.2f}\n' # mean
             f'$\\sigma$ = {np.std(y):.2f}\n' # standard deviation
             f'$r$ = {np.corrcoef(x, y)[0][1]:.2f}') # correlation coefficient
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5) # legen box attributes
    ax.text(0.95, 0.07, stats, fontsize=9, bbox=bbox, # text location, font size, include box 
            transform=ax.transAxes, horizontalalignment='right') # text location

plt.show()
""""
<br />
Comment the resulting visualization!
<br />
<br />
<br />
"""
## References
#Anscombe, F. J. (1973). "Graphs in Statistical Analysis". American Statistician. 27 (1): 17–21. doi:10.1080/00031305.1973.10478966.

"""Anscombe's quartet. https://matplotlib.org/stable/gallery/specialty_plots/anscombe.html

   The use of the following functions, methods, classes and modules is shown
   in this example:

   - `matplotlib.axes.Axes.axline` / `matplotlib.pyplot.axline`
   - `matplotlib.axes.Axes.text` / `matplotlib.pyplot.text`
   - `matplotlib.axes.Axes.tick_params` / matplotlib.pyplot.tick_params"""



