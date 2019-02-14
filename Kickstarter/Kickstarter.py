#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install matplotlib')
get_ipython().system('pip install plotly')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install joblib')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
import inspect
from joblib import dump, load
import random
import os


# In[3]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,                             mean_squared_error, roc_auc_score

from sklearn import svm


# In[4]:


import plotly.tools as tls
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import warnings


# In[5]:


np.random.seed(2000)


# In[6]:


notebook_dir = os.getcwd()
notebook_dir


# In[7]:


df = pd.read_csv("../Kickstarter/ks-projects-201801.csv",
                 encoding='ISO-8859-1',
                 )


# In[8]:


df.head(10)


# In[9]:


df.info()


# In[10]:


df.isna().sum()


# In[11]:


### exclude NA columns
df = df.iloc[:,0:15]


# In[12]:


df['project_length'] = (pd.to_datetime(df['deadline']) - pd.to_datetime(df['launched'])).dt.days + 1


# In[13]:


df.head(5)


# In[14]:


df.shape


# In[15]:


df['state'].value_counts()


# In[16]:


df['main_category'].value_counts()


# In[17]:


# Only consider failed or successful projects
# Exclude suspended, cancelled, and other misc states
df = df[(df['state'] == "failed") | (df['state'] == "successful")]


# In[18]:


# Select proper features for modeling
# Exclude id, name, dates, other features that are not needed in modeling
df_features = df[['main_category', 'currency', 'state', 'backers', 'country', 'usd_pledged_real', 'usd_goal_real', 'project_length']]


# In[19]:


df_features['state'].value_counts()


# In[20]:


df_features.info()


# In[21]:


# Modify our dependent variable to 0 or 1
mapping = {'failed': 0, 'successful': 1}
df_features = df_features.replace({'state': mapping})


# In[22]:


df_features['state'] = pd.to_numeric(df_features['state'], errors='coerce')


# In[23]:


# Categorial columns to numerical using dummy variables
df_features = pd.get_dummies(df_features)


# In[24]:


df_features.head(5)


# In[25]:


# Split the data to train and test
df_train, df_valid = train_test_split(df_features,
                                      test_size = 0.25,
                                      random_state=2018)


# In[26]:


df_train['state'].value_counts()


# In[27]:


df_valid['state'].value_counts()


# In[28]:


y = df_train['state']
X = df_train.drop(columns=['state'])


# In[29]:


X.shape


# In[30]:


y.shape


# In[31]:


inspect.signature(RandomForestClassifier)


# In[32]:


### --- Step 1: Specify different number of trees in forest, to determine
###             how many to use based on leveling-off of OOB error:
n_trees = [50, 100, 250, 500, 1000, 1500]


# In[33]:


### --- Step 2: Create dictionary to save-off each estimated RF model:
rf_dict = dict.fromkeys(n_trees)


# In[34]:


for num in n_trees:
    print(num)
    ### --- Step 3: Specify RF model to estimate:
    rf = RandomForestClassifier(n_estimators=num,
                                min_samples_leaf=30,
                                oob_score=True,
                                random_state=2018,
                                class_weight='balanced',
                                verbose=1)
    ### --- Step 4: Estimate RF model and save estimated model:
    rf.fit(X, y)
    rf_dict[num] = rf


# In[35]:



### --- Save-off model:
# Specify location and name of object to contain estimated model:
model_object_path = os.path.join(notebook_dir, 'rf.joblib')
# Save estimated model to specified location:
dump(rf_dict, model_object_path) 


# In[36]:


# Compute OOB error 
oob_error_list = [None] * len(n_trees)

# Find OOB error for each forest size: 1000 is the best number
for i in range(len(n_trees)):
    oob_error_list[i] = 1 - rf_dict[n_trees[i]].oob_score_
else:
    # Visulaize result:
    plt.plot(n_trees, oob_error_list, 'bo',
             n_trees, oob_error_list, 'k')


# In[37]:


# Feature importance plot
top_num = 20
forest = rf_dict[1000]
importances = forest.feature_importances_

# Sort in decreasing order:
indices = np.argsort(importances)[::-1]    


# In[38]:


len(importances)


# In[39]:


np.array(list(X))[indices[0:top_num]]


# In[40]:


# Plot the feature importances of the forest
ax = plt.gca()
plt.title(f"Top {top_num} feature importances")
plt.bar(range(top_num), importances[indices[0:top_num]])
plt.xticks(range(top_num))
ax.set_xticklabels(np.array(list(X))[indices[0:top_num]], rotation = 90)
ax.set_xlabel("Features")
ax.set_ylabel("Feature Importance")
plt.show()


# In[42]:


y_pred_train = forest.predict(X)
y_pred_train[0:5]


# In[43]:


conf_mat = confusion_matrix(y_true=y,
                            y_pred=y_pred_train)
conf_mat


# In[44]:


class_names = ['failed','successful']


# In[45]:


conf_df = pd.DataFrame(conf_mat, class_names, class_names)
conf_df


# In[46]:


conf_df_pct = conf_df/conf_df.sum(axis=1)
round(conf_df_pct, 2)
# Very Successful results


# In[47]:


# Class-level performance: 0.9803
f1_score(y_true=y,
         y_pred=y_pred_train,
         average='macro')


# In[48]:


# Overall performance across all classes: 0.9809
f1_score(y_true=y,
         y_pred=y_pred_train,
         average='micro')


# In[49]:


# Alternate model: Support Vector Machine
C = 1.0
#svm_dict = dict.fromkeys(['linear', 'LinearSVC', 'rbf', 'poly'])
svm_linear = svm.SVC(kernel='linear', C=C)
#svm_LinearSVC = svm.LinearSVC(C=C)
#svm_rbf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
#svm_poly = svm.SVC(kernel='poly', degree=3, C=C)
svm_linear.fit(X,y)
#svm_LinearSVC.fit(X,y)
#svm_rbf.fit(X,y)
#svm_poly.fit(X,y)

#svm = [svm_linear, svm_LinearSVC, svm_rbf, svm_poly]
#svm_dict['linear'] = svm_linear
#svm_dict['LinearSVC'] = svm_LinearSVC
#svm_dict['rbf'] = svm_rbf
#svm_dict['poly'] = svm_poly


# In[327]:


# Compute OOB error per
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
oob_error_list = [None] * 4

# Find OOB error for each forest size:
for i in range(4):
    oob_error_list[i] = 1 - svm[i].oob_score_
else:
    # Visulaize result:
    plt.plot(np.array(list(svm)), oob_error_list, 'bo',
             np.array(list(svm)), oob_error_list, 'k')


# In[50]:


y_pred_train_svm = svm_linear.predict(X)
y_pred_train_svm[0:5]


# In[51]:


conf_mat_svm = confusion_matrix(y_true=y,
                            y_pred=y_pred_train_svm)
conf_mat_svm


# In[52]:


conf_df_svm = pd.DataFrame(conf_mat_svm, class_names, class_names)
conf_df_svm


# In[53]:


conf_df_pct_svm = conf_df_svm/conf_df_svm.sum(axis=1)
round(conf_df_pct_svm, 2)
# Very Successful results


# In[54]:


# Class-level performance: 0.9998
f1_score(y_true=y,
         y_pred=y_pred_train_svm,
         average='macro')


# In[55]:


# Overall performance across all classes: 0.99998
f1_score(y_true=y,
         y_pred=y_pred_train_svm,
         average='micro')


# In[ ]:




