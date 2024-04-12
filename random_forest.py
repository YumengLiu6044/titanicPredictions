#!/usr/bin/env python
# coding: utf-8

# In[110]:


from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# In[111]:


features = ['Age', 'SibSp', 'Parch', 'Pclass', 'Fare', 'Embarked', 'Sex']

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# In[112]:


# Data cleaning
train_data['Age'].fillna(train_data.Age.mean(), inplace=True)
train_data = train_data[train_data['Embarked'].notna()]

test_data['Age'].fillna(test_data.Age.mean(), inplace=True)
test_data['Embarked'].fillna('No port', inplace=True)
test_data['Fare'].fillna(test_data.Fare.mean(), inplace=True)
test_data.info()


# In[113]:


y = train_data['Survived']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])


# In[114]:


rfc = RandomForestClassifier(n_estimators=100, max_depth=len(features), random_state=1)
rfc.fit(X, y)
y_pred = rfc.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})
output.to_csv('submission.csv', index=False)


# In[114]:




