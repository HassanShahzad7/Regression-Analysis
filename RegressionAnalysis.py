from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

boston = load_boston()
df = pd.DataFrame(boston.data, columns = boston['feature_names'])

df['Price'] = boston.target
new_x = df.drop('Price', axis = 1)
new_y = df['Price']

x_train, x_test, y_train, y_test = train_test_split(new_x, new_y, test_size = 0.3, random_state = 37)
lr = LinearRegression()
rr = Ridge(alpha = 100)
la = Lasso(alpha = 0.001)

lr.fit(x_train, y_train)
rr.fit(x_train, y_train)
la.fit(x_train, y_train)

lr_score = lr.score(x_test, y_test)
rr_score = rr.score(x_test, y_test)
la_score = la.score(x_test, y_test)

lr_score1 = lr.score(x_train, y_train)
rr_score1 = rr.score(x_train, y_train)
la_score1 = la.score(x_train, y_train)

coeff_usedrr = np.sum(rr.coef_ != 0)
coeff_usedla = np.sum(la.coef_ != 0)

print('Training score Linear Regression: ',lr_score1)
print('Testing score Linear Regression: ',lr_score)
print('\nTesting score Ridge Regression: ',rr_score)
print('Training score Ridge Regression: ',rr_score1)
print('\nTraining score Lasso Regression: ',la_score1)
print('Testing score Lasso Regression: ',la_score)

plt.figure()
training = [lr_score, rr_score, la_score]
testing = [lr_score1, rr_score1, la_score1]


plt.scatter(lr_score, lr_score1)
plt.title('Linear Regression')
plt.xlabel('Training Score')
plt.ylabel('Testing Score')
plt.figure()
plt.scatter(rr_score, rr_score1)
plt.title('Ridge Regression')
plt.xlabel('Training Score')
plt.ylabel('Testing Score')
plt.figure()
plt.scatter(la_score, la_score1)
plt.title('Lasso Regression')
plt.xlabel('Training Score')
plt.ylabel('Testing Score')
plt.show()
