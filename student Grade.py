#!/usr/bin/env python
# coding: utf-8

# In[274]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# In[275]:


student = pd.read_csv('Grades.csv')


# In[276]:


student.head()


# In[277]:


# converting missing values
student.isnull().sum()


# In[278]:


# checking for duplicates
student.duplicated()


# In[279]:


# Identify categorical columns (assuming 'student' is your DataFrame)
categorical_columns = student.select_dtypes(include=['object']).columns


# In[280]:


# One-hot encode categorical columns
student_encoded = pd.get_dummies(student, columns=categorical_columns, drop_first=True)


# In[281]:


# Separate features (X) and target variable (y)
X = student_encoded.drop('CGPA', axis=1)  # Exclude the target variable
y = student_encoded['CGPA']


# In[282]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[283]:


# Initialize the linear regression model
model = LinearRegression()


# Train the model
model.fit(X_train, Y_train)


# In[284]:


# Make predictions on the test set
Y_pred = model.predict(X_test)


# In[285]:


# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




