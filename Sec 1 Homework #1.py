#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import statsmodels.api as sm


# # 1.) Import Data from FRED

# In[55]:


data = pd.read_csv("TaylorRuleData.csv", index_col = 0)


# In[56]:


data.index = pd.to_datetime(data.index)


# In[57]:


data = data.dropna()


# In[58]:


data.head()


# # 2.) Do Not Randomize, split your data into Train, Test Holdout

# In[59]:


split1 = int(len(data) * .6)
split2 = int(len(data) * .9)
data_in = data[:split1]
data_out = data[split1:split2]
data_hold = data[split2:]


# In[60]:


X_in = data_in.iloc[:,1:]
y_in = data_in.iloc[:,0]
X_out = data_out.iloc[:,1:]
y_out = data_out.iloc[:,0]
X_hold = data_hold.iloc[:,1:]
y_hold = data_hold.iloc[:,0]


# In[61]:


# Add Constants
X_in = sm.add_constant(X_in)
X_out = sm.add_constant(X_out)
X_hold = sm.add_constant(X_hold)


# # 3.) Build a model that regresses FF~Unemp, HousingStarts, Inflation

# In[62]:


model1 = sm.OLS(y_in, X_in).fit()


# # 4.) Recreate the graph fro your model

# In[63]:


import matplotlib.pyplot as plt


# In[64]:


plt.figure(figsize = (12,5))

plt.plot(y_in)
plt.plot(y_out)
plt.plot(model1.predict(X_in))
plt.plot(model1.predict(X_out))

plt.ylabel("Fed Funds")
plt.xlabel("Time")
plt.title("Visualizing Model Accuracy")
plt.legend([])
plt.grid()
plt.show()


# ## "All Models are wrong but some are useful" - 1976 George Box

# # 5.) What are the in/out of sample MSEs

# In[65]:


from sklearn.metrics import mean_squared_error


# In[66]:


in_mse_1 = mean_squared_error(y_in, model1.predict(X_in))
out_mse_1 = mean_squared_error(y_out, model1.predict(X_out))


# In[46]:


print("Insample MSE : ", in_mse_1)
print("Outsample MSE : ", out_mse_1)


# # 6.) Using a for loop. Repeat 3,4,5 for polynomial degrees 1,2,3

# In[47]:


from sklearn.preprocessing import PolynomialFeatures


# In[48]:


max_degrees = 3


# In[67]:


for degrees in range(1,max_degrees + 1):
    print("Degrees:", degrees)
    poly = PolynomialFeatures(degree = degrees)
    X_in_poly = poly.fit_transform(X_in)
    X_out_poly = poly.transform(X_out)
    
    model1 = sm.OLS(y_in, X_in_poly).fit()
    
    plt.figure(figsize = (12,5))
    in_preds = model1.predict(X_in_poly)
    in_preds = pd.DataFrame(in_preds, index = y_in.index)
    
    out_preds = model1.predict(X_out_poly)
    out_preds = pd.DataFrame(out_preds, index = y_out.index)
    
    plt.plot(y_in)
    plt.plot(y_out)
    plt.plot(in_preds)
    plt.plot(out_preds)

    plt.ylabel("Fed Funds")
    plt.xlabel("Time")
    plt.title("Visualizing Model Accuracy")
    plt.legend([])
    plt.grid()
    plt.show()
    
    # Q5
    in_mse_1 = mean_squared_error(y_in, model1.predict(X_in_poly))
    out_mse_1 = mean_squared_error(y_out, model1.predict(X_out_poly))
    print("Insample MSE : ", in_mse_1)
    print("Outsample MSE : ", out_mse_1)


# # 7.) State your observations :

# 

# The results reinforce the concept articulated by George Box that "all models are wrong, but some are useful." The simpler linear model appears to be more useful when considering the ability to generalize, despite its larger in-sample error compared to higher-degree models.
# 
# It's important to balance model complexity with the ability to generalize to new data. Lower complexity models may perform better on unseen data, even if they don't capture all the nuances in the training set.
# 
# Based on the MSE values provided, degree 1 may be the preferred model for predicting the Federal Funds rate given this dataset, as it provides a reasonable trade-off between fit and generalizability. Higher-degree polynomial models, while reducing the in-sample error, do not generalize well and result in high out-of-sample error, indicative of overfitting.
