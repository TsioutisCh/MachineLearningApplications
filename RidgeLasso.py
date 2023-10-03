#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sklearn  
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


np.random.seed(20) #random seed - replicate
X=np.random.randn(2000,5) #2000 random points - each one having a dim of 5

X=scale(X)#cheating a bit


#true function is 2*x1 + 0.5*x3 + 2*x4
yclean=2*X[:,1] + 0.5*X[:,3] + 2*X[:,4]
y=yclean

#y=y+np.random.normal(0,np.std(y)/2,y.shape)
y=y+np.random.normal(0,np.std(y)*3,y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def regplot(L,coefs,errors,method):
  plt.figure(figsize=(20, 6))

  plt.subplot(121)
  ax = plt.gca()
  ax.plot(L, coefs)
  ax.set_xscale('log')
  plt.xlabel('alpha')
  plt.ylabel('weights')
  plt.title(method + ' coefficients as a function of the regularization')
  plt.axis('tight')

  plt.subplot(122)
  ax = plt.gca()
  ax.plot(L, errors)
  ax.set_xscale('log')
  plt.xlabel('alpha')
  plt.ylabel('error')
  plt.title(method+' error as a function of the regularization')
  plt.axis('tight')

  plt.show()


# In[ ]:


#true function is 2*x1 + 0.5*x3 + 2*x4
modelLR=LinearRegression(fit_intercept=True)
modelLR.fit(X_train, y_train)
y_pred=modelLR.predict(X_test)
print("Parameters:", modelLR.coef_, modelLR.intercept_,"Error=",mean_squared_error(y_test,y_pred))


# In[ ]:


#lasso - l1 regularization
# mse + |parameters|_1
coefs = []
errors = []

#lamda
L=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
for alphav in L:
  alphav
  model = Lasso(alpha=alphav)
  model.fit(X_train,y_train)
  y_pred=model.predict(X_test)
  coefs.append(model.coef_)
  errors.append(mean_squared_error(y_test,y_pred))

  print("Parameters:", model.coef_, model.intercept_,alphav, mean_squared_error(y_test,y_pred))
#true function is 2*x1 + 0.5*x3 + 2*x4


# In[ ]:


regplot(L,coefs,errors,"Lasso")


# In[ ]:


#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
#l2 regularization + mse


coefs = []
errors = []

#true function is 2*x1 + 0.5*x3 + 2*x4
L=np.array([0.01, 0.1,  0.2, 0.25, 0.3, 0.35, 0.5,10,100,200]) #lambda hyperpameter
for alphav in L:
  model = Ridge(alpha=alphav)
  model.fit(X_train,y_train)
  y_pred=model.predict(X_test)
  coefs.append(model.coef_)
  errors.append(mean_squared_error(y_test,y_pred))  

  print("Parameters:", model.coef_, model.intercept_,alphav, mean_squared_error(y_test,y_pred))


# In[ ]:


regplot(L,coefs,errors,"Ridge")


# In[ ]:




