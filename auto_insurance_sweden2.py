# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 23:59:24 2020

@author: ADMIN
"""

import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm
import warnings
import statsmodels.formula.api as smf

from mpl_toolkits.mplot3d import axes3d
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split


data = pd.read_csv(r'C:\Users\ADMIN\Desktop\data_sci\regressions\auto_insurance_sweden.csv')

data.info()
data.describe()


data_new= data.rename(columns={'108':'X','392.5':'Y'})

X = data['108'].values
Y = data['392.5'].values

print(data_new)

data_new_corr = data_new.corr()

print(data_new_corr)

data_new.describe()


#Simple linear regression

#Least Squares Fit
sns.regplot(data_new.X, data_new.Y,order=1,ci=None,scatter_kws={'color':'r','s':40})
plt.xlim(-50,150)
plt.ylim(0,450);


"""REsidual sum of squares"""
#meshgrid
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
h = plt.contourf(x,y,z)
plt.show()


#Regression coefficients
regr = skl_lm.LinearRegression()
x=scale(data_new.X, with_mean=True, with_std=False).reshape(-1,1)
y=data_new.Y


regr.fit(x,y)
print(regr.intercept_)###Beta_0
print(regr.coef_)##Beta_1

B0 = np.linspace(regr.intercept_-2, regr.intercept_+2, 62)
B1 = np.linspace(regr.coef_-0.02, regr.coef_+0.02, 62)
xx, yy = np.meshgrid(B0, B1, indexing='xy')
Z = np.zeros((B0.size,B1.size))

# Calculate Z-values (RSS) based on grid of coefficients

for (i,j),v in np.ndenumerate(Z):
   
    Z[i,j] =((y - (xx[i,j]+X.ravel()*yy[i,j]))**2).sum()/1000

# Minimized RSS
min_RSS = r'$\beta_0$, $\beta_1$ for minimized RSS'
min_rss = np.sum((regr.intercept_+regr.coef_*X - y.values.reshape(-1,1))**2)/1000
min_rss

fig = plt.figure(figsize=(10,6))
fig.suptitle('RSS - Regression coefficients', fontsize=20)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Left plot
CS = ax1.contour(xx, yy, Z, cmap=plt.cm.Set1, levels=[2.15, 2.2, 2.3, 2.5, 3])
ax1.scatter(regr.intercept_, regr.coef_[0], c='r', label=min_RSS)
ax1.clabel(CS, inline=True, fontsize=10, fmt='%1.1f')

# Right plot
ax2.plot_surface(xx, yy, Z, rstride=3, cstride=3, alpha=0.3)
ax2.contour(xx, yy, Z, zdir='z', offset=Z.min(), cmap=plt.cm.Set1,
            alpha=0.4, levels=[2.15, 2.2, 2.3, 2.5, 3])
ax2.scatter3D(regr.intercept_, regr.coef_[0], min_rss, c='r', label=min_RSS)
ax2.set_zlabel('RSS')
ax2.set_zlim(Z.min(),Z.max())
ax2.set_ylim(0.02,0.07)


# settings common to both plots
for ax in fig.axes:
    ax.set_xlabel(r'$\beta_0$', fontsize=17)
    ax.set_ylabel(r'$\beta_1$', fontsize=17)
    ax.set_yticks([0.03,0.04,0.05,0.06])
    ax.legend()
'''

subplot()
Either a 3-digit integer or three separate integers describing the position of the subplot.
If the three integers are nrows, ncols, and index in order,
the subplot will take the index position on a grid with nrows rows and ncols columns.
index starts at 1 in the upper left corner and increases to the right.

pos is a three digit integer, where the first digit is the number of rows, the second the number of columns,
and the third the index of the subplot. i.e. fig.add_subplot(235) is the same as fig.add_subplot(2, 3, 5).
 Note that all integers must be less than 10 for this form to work.
 
'''



##Confidence interval

est = smf.ols('X ~ Y', data_new).fit()
est.summary().tables[1]


# RSS with regression coefficients
((data_new.X - (est.params[0] + est.params[1]*data_new.X))**2).sum()/1000

#####Using sk-learn
regr = skl_lm.LinearRegression()

X = data_new.Y.values.reshape(-1,1)
y = data_new.X

regr.fit(X,y)
print(regr.intercept_)
print(regr.coef_)

X_pred = regr.predict(X)
r2_score(y, X_pred)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=42, shuffle = True)


def graphics(y, X_pred):
   
    # scatter
    plt.figure(figsize=(12, 6))

    plt.plot(y,y)
    plt.scatter(X_pred,y, c='r', marker='o')
    plt.legend(['Actual','Predicted'])
    plt.grid(ls='-.', lw=0.2, c='k');
   
    # distplot
    plt.figure(figsize=(12, 6))    
    sns.distplot(y)
    sns.distplot(X_pred)
    plt.legend(['Actual','Predicted'])
    plt.grid(ls='-.', lw=0.2, c='k')

from sklearn.linear_model import LinearRegression

lR = LinearRegression()

lR.fit(X_train,y_train)

pred_lR = lR.predict(X_test)

info(y_test, pred_lR, 'LinearRegression')

graphics(y_test, pred_lR)


#meanX and Y
mean_x = np.mean(X)
mean_y = np. mean(Y)

#Total no of values
n = len(X)


# Using the formula to calculate m and c
numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
    m = numer / denom
    c = mean_y - (m * mean_x)

# Print coefficients
print(m,c)


# Plotting Values and Regression Line

max_x = np.max(X) + 100
min_x = np.min(X) - 100
# Calculating line values x and y
x = np.linspace(min_x, max_x, 500)
y = c + m * x 
 
# Ploting Line
plt.plot(x, y, color='r', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='r', label='Scatter Plot')
 
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(X,Y)
plt.show()
