#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
import inspect
from joblib import dump, load
import random
import os
import dask.dataframe as dd
import requests
from time import time
from datetime import date, datetime


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, roc_auc_score
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

import plotly.tools as tls
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

np.random.seed(2000)
notebook_dir = os.getcwd()

# used dask to read csv to speed up
t0 = time()
df = dd.read_csv("../Kickstarter/ks-projects-201801.csv",
                 encoding='ISO-8859-1'
                 )
t1 = time()
print((t1 - t0), 'seconds')

# use pandas functions
df = df.compute()

df.head(10)

df.info()

df.isna().sum()

df[df['name'].isna()].head()

# In[13]:


df[df['usd pledged'].isna()].head()

# In[14]:



# Only consider failed or successful projects
# Exclude suspended, cancelled, and other misc states
df = df[(df['state'] == "failed") | (df['state'] == "successful")]


# changed launched data from datetime to date in the csv file
# also utilized apply function as well as
# datetime.strptime.date() which is faster than pd.to_datetime to reduce the time by 4 times
t0 = time()
df['project_length'] = df.apply(lambda row: (datetime.strptime((row['deadline']),
                                                               "%m/%d/%Y").date() - datetime.strptime((row['launched']),
                                                                                                      "%m/%d/%Y").date()).days + 1,
                                axis=1)
t1 = time()
print((t1 - t0), 'seconds')

df[['deadline', 'launched', 'project_length']].head(5)
# use better index for speeding
df.set_index('ID', inplace=True)
# drop unneccesary columns
to_drop = ['name', 'category', 'deadline', 'launched', 'goal', 'pledged', 'usd pledged']
df.drop(to_drop, inplace=True, axis=1)

# Select proper features for modeling
# Exclude id, name, dates, other features that are not needed in modeling
# We drop usd_pledged_real because the model would be perfectly predicting outcomes if it knows both pledged and goal values
df_features = df[['main_category', 'state', 'country', 'usd_goal_real', 'project_length']]
# Modify our dependent variable to 0 or 1
mapping = {'failed': 0, 'successful': 1}
df_features = df_features.replace({'state': mapping})

df_features['state'] = pd.to_numeric(df_features['state'], errors='coerce')
# Categorial columns to numerical using dummy variables
df_features = pd.get_dummies(df_features)

# Split the data to train and test
df_train, df_valid = train_test_split(df_features,
                                      test_size=0.25,
                                      random_state=2018)
df_train['state'].value_counts()
df_valid['state'].value_counts()

y = df_train['state']
X = df_train.drop(columns=['state'])
y_test = df_valid['state']
X_test = df_valid.drop(columns=['state'])

### --- Step 1: Specify different number of trees in forest, to determine
###             how many to use based on leveling-off of OOB error:
n_trees = [50, 100, 250, 500, 1000, 1500]

### --- Step 2: Create dictionary to save-off each estimated RF model:
rf_dict = dict.fromkeys(n_trees)

t0 = time()

for num in n_trees:
    print(num)
    ### --- Step 3: Specify RF model to estimate:
    rf = RandomForestClassifier(n_estimators=num,
                                min_samples_leaf=30,
                                oob_score=True,
                                random_state=2019,
                                class_weight='balanced',
                                verbose=1,
                                n_jobs=4)
    ### --- Step 4: Estimate RF model and save estimated model:
    rf.fit(X, y)
    rf_dict[num] = rf

t1 = time()
print((t1 - t0), 'seconds')

### --- Save-off model:
# Specify location and name of object to contain estimated model:
model_object_path = os.path.join(notebook_dir, 'rf.joblib')
# Save estimated model to specified location:
dump(rf_dict, model_object_path)

# Compute OOB error
oob_error_list = [None] * len(n_trees)

# Find OOB error for each forest size: 1000 is the best number
for i in range(len(n_trees)):
    oob_error_list[i] = 1 - rf_dict[n_trees[i]].oob_score_
else:
    # Visulaize result:
    fig = plt.figure()
    plt.plot(n_trees, oob_error_list, 'bo',
             n_trees, oob_error_list, 'k')
    fig.suptitle('Error vs Number of trees')
    plt.xlabel('number of trees')
    plt.ylabel('OOB error')

### Specify different number of leafs, to determine
### how many to use based on leveling-off of OOB error:
n_leaves = [5, 10, 30, 50, 100, 200]
rf_dict2 = dict.fromkeys(n_leaves)
t0 = time()

for num in n_leaves:
    print(num)
    ### --- Step 3: Specify RF model to estimate:
    rf = RandomForestClassifier(n_estimators=1500,
                                min_samples_leaf=num,
                                oob_score=True,
                                random_state=2019,
                                class_weight='balanced',
                                verbose=1,
                                n_jobs=4)
    ### --- Step 4: Estimate RF model and save estimated model:
    rf.fit(X, y)
    rf_dict2[num] = rf

t1 = time()
print((t1 - t0), 'seconds')

# Compute OOB error
oob_error_list = [None] * len(n_leaves)

# Find OOB error for each forest size: 1000 is the best number
for i in range(len(n_leaves)):
    oob_error_list[i] = 1 - rf_dict2[n_leaves[i]].oob_score_
else:
    # Visulaize result:
    fig = plt.figure()
    plt.plot(n_leaves, oob_error_list, 'bo',
             n_leaves, oob_error_list, 'k')
    fig.suptitle('Error vs Number of leaves')
    plt.xlabel('number of leaves')
    plt.ylabel('OOB error')

# Feature importance plot
top_num = 20
forest = rf_dict[1500]
importances = forest.feature_importances_
# Sort in decreasing order:
indices = np.argsort(importances)[::-1]
len(importances)
np.array(list(X))[indices[0:top_num]]
# Plot the feature importances of the forest
ax = plt.gca()
plt.title(f"Top {top_num} feature importances")
plt.bar(range(top_num), importances[indices[0:top_num]])
plt.xticks(range(top_num))
ax.set_xticklabels(np.array(list(X))[indices[0:top_num]], rotation=90)
ax.set_xlabel("Features")
ax.set_ylabel("Feature Importance")
plt.show()

# Model Validation
y_pred_test = forest.predict(X_test)

conf_mat = confusion_matrix(y_true=y_test,
                            y_pred=y_pred_test)

class_names = ['failed', 'successful']
conf_df = pd.DataFrame(conf_mat, class_names, class_names)
conf_df_pct = conf_df / conf_df.sum(axis=0)
round(conf_df_pct * 100, 1)
# Fairly Successful results

# Class-level performance:
f1_score(y_true=y_test,
         y_pred=y_pred_test,
         average='macro')
# Overall performance across all classes:
f1_score(y_true=y_test,
         y_pred=y_pred_test,
         average='micro')

