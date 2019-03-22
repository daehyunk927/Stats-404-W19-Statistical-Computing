#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install matplotlib')
get_ipython().system('pip install plotly')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install joblib')
get_ipython().system('pip install dask[dataframe]')


# In[2]:


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


# In[3]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,                             mean_squared_error, roc_auc_score

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier


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


# used dask to read csv to speed up
t0 = time()
df = dd.read_csv("../Kickstarter/ks-projects-201801.csv",
                 encoding='ISO-8859-1'
                 )
t1 = time()
print ((t1 - t0) , 'seconds')


# In[8]:


# use pandas functions
df = df.compute()


# In[9]:


df.head(10)


# In[10]:


df.info()


# In[11]:


df.isna().sum()


# In[12]:


df[df['name'].isna()].head()


# In[13]:


df[df['usd pledged'].isna()].head()


# In[14]:


# EDA
percentual_sucess = round(df["state"].value_counts() / len(df["state"]) * 100,2)

print("State Percent in %: ")
print(percentual_sucess)


# In[15]:


state = round(df["state"].value_counts() / len(df["state"]) * 100,2)

labels = list(state.index)
values = list(state.values)

trace1 = go.Pie(labels=labels, values=values, marker=dict(colors=['red']))

layout = go.Layout(title='State Distribution', legend=dict(orientation="h"));

fig = go.Figure(data=[trace1], layout=layout)
iplot(fig)


# In[16]:


# Only consider failed or successful projects
# Exclude suspended, cancelled, and other misc states
df = df[(df['state'] == "failed") | (df['state'] == "successful")]


# In[17]:


state = round(df["state"].value_counts() / len(df["state"]) * 100,2)

labels = list(state.index)
values = list(state.values)

trace1 = go.Pie(labels=labels, values=values, marker=dict(colors=['red']))

layout = go.Layout(title='State Distribution', legend=dict(orientation="h"));

fig = go.Figure(data=[trace1], layout=layout)
iplot(fig)


# In[18]:


main_colors = dict({'failed': 'rgb(200,50,50)', 'successful': 'rgb(115,50,200)'})


# In[19]:


data = []
annotations = []

rate_success_cat = df[df['state'] == 'successful'].groupby(['main_category']).count()['ID']/ df.groupby(['main_category']).count()['ID'] * 100
rate_failed_cat = df[df['state'] == 'failed'].groupby(['main_category']).count()['ID']/ df.groupby(['main_category']).count()['ID'] * 100
    
rate_success_cat = rate_success_cat.sort_values(ascending=False)
rate_failed_cat = rate_failed_cat.sort_values(ascending=True)

bar_success = go.Bar(
        x=rate_success_cat.index,
        y=rate_success_cat,
        name='successful',
        marker=dict(
            color=main_colors['successful'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )
bar_failed = go.Bar(
        x=rate_failed_cat.index,
        y=rate_failed_cat,
        name='failed',
        marker=dict(
            color=main_colors['failed'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

data = [bar_success, bar_failed]
layout = go.Layout(
    barmode='stack',
    title='% of successful and failed projects by main category',
    autosize=False,
    width=800,
    height=400,
    annotations=annotations
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='main_cat')

#It looks that some categories are more likely to be successful than others. It might be a matter of people's interest or other factors.
#For example, the goal amount in some categories on average may be lower than in others and projects of this category are more likely to succeed. 
#That is why the 'goal amount' by category is the next thing we should look at.


# In[20]:


data = []

goal_success = df[df['state'] == 'successful'].groupby(['main_category'])                    .median()['usd_goal_real'].reindex(rate_success_cat.index)
goal_failed = df[df['state'] == 'failed'].groupby(['main_category'])                    .median()['usd_goal_real'].reindex(rate_success_cat.index)

bar_success = go.Bar(
        x=goal_success.index,
        y=goal_success,
        name='successful',
        marker=dict(
            color=main_colors['successful'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

bar_failed = go.Bar(
        x=goal_failed.index,
        y=goal_failed,
        name='failed',
        marker=dict(
            color=main_colors['failed'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

data = [bar_success, bar_failed]
layout = go.Layout(
    barmode='group',
    title='Median goal of successful and failed projects by main category (in USD)',
    autosize=False,
    width=800,
    height=400
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='median_goal_main_cat')

#We are not looking at 'mean' since single outliers may have an impact we do not want to capture.
#First thing we notice is that the projects with higher goal amount are more likely to fail.
#Another thing is that the most successful categories have relatively low goal amount. However, this dependency is not that strong and it is difficult to rely on this pattern too much.

#We may also assume that some categories are less successful since they have too many 'outliers' with too high goal amounts. That is why we will look at the difference of median goal amounts of failed and successful projects.
#Since $1000 may be a significant sum for one category(eg. Craft) and less significant for another(eg. Technology), we will calculate relative differe


# In[21]:


pleged_failed = df[df['state'] == 'failed']['usd_pledged_real']                        /df[df['state'] == 'failed']['usd_goal_real']*100
data = [go.Histogram(x=pleged_failed, marker=dict(color=main_colors['failed']))]

layout = go.Layout(
    title='% pledged of the goal amount for failed projects',
    autosize=False,
    width=800,
    height=400
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='pleged_failed')
#It seems that the majority of failing projects do not get even several percent of the goal amount before deadline.


# In[22]:


# previous code ~ 70 seconds
#t0 = time()
#df['project_length'] = (pd.to_datetime(df['deadline']) - pd.to_datetime(df['launched'])).dt.days + 1
#t1 = time()
#print ((t1 - t0) , 'seconds')


# In[23]:


# changed launched data from datetime to date in the csv file
# also utilized apply function as well as 
# datetime.strptime.date() which is faster than pd.to_datetime to reduce the time by 4 times
t0 = time()
df['project_length'] = df.apply(lambda row: (datetime.strptime((row['deadline']), "%m/%d/%Y").date()-datetime.strptime((row['launched']), "%m/%d/%Y").date()).days+1, axis=1)
t1 = time()
print ((t1 - t0) , 'seconds')


# In[24]:


df[['deadline','launched','project_length']].head(5)


# In[25]:


data = [go.Histogram(x=df[df['state'] == 'failed']['project_length'], 
                     marker=dict(color=main_colors['failed']),
                     name='failed'),
        go.Histogram(x=df[df['state'] == 'successful']['project_length'], 
                     marker=dict(color=main_colors['successful']),
                     name='successful')]

layout = go.Layout(
    barmode='stack',
    title='Project length distribtuion',
    autosize=False,
    width=800,
    height=400
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='project_length_distribution')


# In[26]:


print('Mean days for failed projects: {0}'
      .format(round(df[df['state'] == 'failed']['project_length'].mean(), 2)))
print('Mean days for successful projects: {0}'
      .format(round(df[df['state'] == 'successful']['project_length'].mean(), 2)))


# In[27]:


# Replacing unknown value to nan
df['country'] = df['country'].replace('N,0"', np.nan)

data = []
total_expected_values = []
annotations = []
shapes = []

rate_success_country = df[df['state'] == 'successful'].groupby(['country']).count()['ID']                / df.groupby(['country']).count()['ID'] * 100
rate_failed_country = df[df['state'] == 'failed'].groupby(['country']).count()['ID']                / df.groupby(['country']).count()['ID'] * 100
    
rate_success_country = rate_success_country.sort_values(ascending=False)
rate_failed_country = rate_failed_country.sort_values(ascending=True)

bar_success = go.Bar(
        x=rate_success_country.index,
        y=rate_success_country,
        name='successful',
        marker=dict(
            color=main_colors['successful'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

bar_failed = go.Bar(
        x=rate_failed_country.index,
        y=rate_failed_country,
        name='failed',
        marker=dict(
            color=main_colors['failed'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

data = [bar_success, bar_failed]
layout = go.Layout(
    barmode='stack',
    title='% of successful and failed projects by country',
    autosize=False,
    width=800,
    height=400,
    annotations=annotations,
    shapes=shapes
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='main_cat')


# In[28]:


df.shape


# In[29]:


# use better index for speeding
df.set_index('ID', inplace=True)


# In[30]:


# drop unneccesary columns
to_drop = ['name', 'category', 'deadline', 'launched', 'goal', 'pledged', 'usd pledged']
df.drop(to_drop, inplace=True, axis=1)


# In[31]:


df.head(10)


# In[32]:


# Select proper features for modeling
# Exclude id, name, dates, other features that are not needed in modeling
# We drop usd_pledged_real because the model would be perfectly predicting outcomes if it knows both pledged and goal values
df_features = df[['main_category', 'state', 'country', 'usd_goal_real', 'project_length']]


# In[33]:


df_features.info()


# In[34]:


# Modify our dependent variable to 0 or 1
mapping = {'failed': 0, 'successful': 1}
df_features = df_features.replace({'state': mapping})


# In[35]:


df_features['state'] = pd.to_numeric(df_features['state'], errors='coerce')


# In[36]:


# Categorial columns to numerical using dummy variables
df_features = pd.get_dummies(df_features)


# In[37]:


df_features.head(5)


# In[38]:


# Split the data to train and test
df_train, df_valid = train_test_split(df_features,
                                      test_size = 0.25,
                                      random_state=2018)


# In[39]:


df_train['state'].value_counts()


# In[40]:


df_valid['state'].value_counts()


# In[41]:


y = df_train['state']
X = df_train.drop(columns=['state'])


# In[42]:


y_test = df_valid['state']
X_test = df_valid.drop(columns=['state'])


# In[43]:


X.shape


# In[44]:


y.shape


# In[45]:


inspect.signature(RandomForestClassifier)


# In[48]:


### --- Step 1: Specify different number of trees in forest, to determine
###             how many to use based on leveling-off of OOB error:
n_trees = [50, 100, 250, 500, 1000, 1500]


# In[49]:


### --- Step 2: Create dictionary to save-off each estimated RF model:
rf_dict = dict.fromkeys(n_trees)


# In[50]:


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
print ((t1 - t0) , 'seconds')


# In[51]:


### --- Save-off model:
# Specify location and name of object to contain estimated model:
model_object_path = os.path.join(notebook_dir, 'rf.joblib')
# Save estimated model to specified location:
dump(rf_dict, model_object_path) 


# In[52]:


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


# In[53]:


### Specify different number of leafs, to determine
### how many to use based on leveling-off of OOB error:
n_leaves = [5,10,30,50,100,200]


# In[54]:


rf_dict2 = dict.fromkeys(n_leaves)


# In[60]:


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
print ((t1 - t0) , 'seconds')


# In[61]:


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


# In[62]:


# Feature importance plot
top_num = 20
forest = rf_dict[1500]
importances = forest.feature_importances_

# Sort in decreasing order:
indices = np.argsort(importances)[::-1]    


# In[63]:


len(importances)


# In[64]:


np.array(list(X))[indices[0:top_num]]


# In[65]:


# Plot the feature importances of the forest
ax = plt.gca()
plt.title(f"Top {top_num} feature importances")
plt.bar(range(top_num), importances[indices[0:top_num]])
plt.xticks(range(top_num))
ax.set_xticklabels(np.array(list(X))[indices[0:top_num]], rotation = 90)
ax.set_xlabel("Features")
ax.set_ylabel("Feature Importance")
plt.show()


# In[66]:


y_pred_train = forest.predict(X)
y_pred_train[0:5]


# In[67]:


conf_mat = confusion_matrix(y_true=y,
                            y_pred=y_pred_train)
conf_mat


# In[68]:


class_names = ['failed','successful']


# In[69]:


conf_df = pd.DataFrame(conf_mat, class_names, class_names)
conf_df


# In[70]:


conf_df_pct = conf_df/conf_df.sum(axis=0)
round(conf_df_pct*100, 1)
# Very Successful results


# In[71]:


# Class-level performance: 0.9803
f1_score(y_true=y,
         y_pred=y_pred_train,
         average='macro')


# In[72]:


# Overall performance across all classes: 0.9809
f1_score(y_true=y,
         y_pred=y_pred_train,
         average='micro')


# In[73]:


y_pred_test = forest.predict(X_test)


# In[74]:


conf_mat = confusion_matrix(y_true=y_test,
                            y_pred=y_pred_test)
conf_mat


# In[75]:


class_names = ['failed','successful']
conf_df = pd.DataFrame(conf_mat, class_names, class_names)
conf_df


# In[76]:


conf_df_pct = conf_df/conf_df.sum(axis=0)
round(conf_df_pct*100, 1)
# Very Successful results


# In[77]:


# Class-level performance: 0.978
f1_score(y_true=y_test,
         y_pred=y_pred_test,
         average='macro')


# In[78]:


# Overall performance across all classes: 0.979
f1_score(y_true=y_test,
         y_pred=y_pred_test,
         average='micro')


# In[79]:


inspect.signature(OneVsRestClassifier)


# In[ ]:


# Alternate model: Support Vector Machine
# Used OnvVsRestClassifier to run svm in parallel
t0 = time()
C = 1.0
#svm_linear = svm.SVC(kernel='linear', C=C)
clf = OneVsRestClassifier(svm.SVC(kernel='linear', C=C), n_jobs=4)
#svm_LinearSVC = svm.LinearSVC(C=C)
#svm_rbf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
#svm_poly = svm.SVC(kernel='poly', degree=3, C=C)
#svm_linear.fit(X,y)
#svm_LinearSVC.fit(X,y)
#svm_rbf.fit(X,y)
#svm_poly.fit(X,y)
clf.fit(X,y)
t1 = time()
print ((t1 - t0) , 'seconds')

#571 from rf vs 3568 from svm


# In[ ]:


y_pred_train_svm = clf.predict(X)
y_pred_train_svm[0:5]


# In[77]:


conf_mat_svm = confusion_matrix(y_true=y,
                            y_pred=y_pred_train_svm)
conf_mat_svm


# In[78]:


conf_df_svm = pd.DataFrame(conf_mat_svm, class_names, class_names)
conf_df_svm


# In[79]:


conf_df_pct_svm = conf_df_svm/conf_df_svm.sum(axis=1)
round(conf_df_pct_svm, 2)
# Very Successful results


# In[80]:


# Overall performance across all classes: 0.99998
f1_score(y_true=y,
         y_pred=y_pred_train_svm,
         average='micro')


# In[81]:


# Overall performance across all classes: 0.99998
f1_score(y_true=y,
         y_pred=y_pred_train_svm,
         average='macro')


# In[108]:


y_pred_test_svm = clf.predict(X_test)
y_pred_test_svm[0:5]


# In[109]:


conf_mat_svm = confusion_matrix(y_true=y_test,
                            y_pred=y_pred_test_svm)
conf_mat_svm


# In[110]:


conf_df_svm = pd.DataFrame(conf_mat_svm, class_names, class_names)
conf_df_svm


# In[111]:


conf_df_pct_svm = conf_df_svm/conf_df_svm.sum(axis=0)
round(conf_df_pct_svm, 5)
# Very Successful results


# In[112]:


# Overall performance across all classes: 0.9998
f1_score(y_true=y_test,
         y_pred=y_pred_test_svm,
         average='micro')


# In[113]:


# Overall performance across all classes: 0.99997
f1_score(y_true=y_test,
         y_pred=y_pred_test_svm,
         average='macro')

