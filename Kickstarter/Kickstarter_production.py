#!/usr/bin/env python
# coding: utf-8
import os
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
import dask.dataframe as dd


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

np.random.seed(2000)
notebook_dir = os.getcwd()

# ------------------------------------
# Part 1: Data Processing
# ------------------------------------

# Load the data file using dask
df = dd.read_csv("../Kickstarter/ks-projects-201801.csv",
                 encoding='ISO-8859-1'
                 )
# use pandas functions instead of dask
df = df.compute()

# Only consider failed or successful projects
# Exclude suspended, cancelled, and other misc states


def drop_unused_state(data):
    data = data[(data['state'] == "failed") | (data['state'] == "successful")]
    return data


df = drop_unused_state(df)

# Find the length of each project by calculating the number of days
# between launched date and deadline date


def calculate_project_length(data):
    data['project_length'] = data.apply(lambda row:
                                    (datetime.strptime((row['deadline']),
                                                       "%m/%d/%Y").date()
                                     - datetime.strptime((row['launched']),
                                                         "%m/%d/%Y").date()).days + 1,
                                    axis=1)
    return data


df = calculate_project_length(df)

# use better index for speeding
df.set_index('ID', inplace=True)
# drop unnecessary columns
to_drop = ['name', 'category', 'deadline', 'launched',
           'goal', 'pledged', 'usd pledged']


def drop_unnecesaary_columns(to_drop, data):
    data.drop(to_drop, inplace=True, axis=1)
    return data


df = drop_unnecesaary_columns(to_drop, df)

# Select proper features for modeling
# Exclude id, name, dates, other features that are not needed in modeling
# We drop usd_pledged_real because the model would be perfectly
# predicting outcomes if it knows both pledged and goal values
df_features = df[['main_category', 'state', 'country',
                  'usd_goal_real', 'project_length']]


# Create Dummy variables for all the categorical variables
Mapping = {'failed': 0, 'successful': 1}


def change_categorical_to_numerical(mapping, data):
    data = data.replace({'state': mapping})
    data['state'] = pd.to_numeric(data['state'], errors='coerce')
    data = pd.get_dummies(data)

    return data


df_features = change_categorical_to_numerical(Mapping, df_features)

# Split the data to train and test
df_train, df_valid = train_test_split(df_features,
                                      test_size=0.25,
                                      random_state=2018)

# Set up training and testing variable sets
y = df_train['state']
X = df_train.drop(columns=['state'])
y_test = df_valid['state']
X_test = df_valid.drop(columns=['state'])


# -------------------------------------------
# Part 2: Model Estimation
# -------------------------------------------

# --- Step 1: Specify different number of trees in forest, to determine
#             how many to use based on leveling-off of OOB error:
#n_trees = [50, 100, 250, 500, 1000, 1500]
n_trees = [50, 100]

# --- Step 2: Create dictionary to save-off each estimated RF model:
rf_dict = dict.fromkeys(n_trees)

for num in n_trees:
    print(num)
    # --- Step 3: Specify RF model to estimate:
    rf = RandomForestClassifier(n_estimators=num,
                                min_samples_leaf=30,
                                oob_score=True,
                                random_state=2019,
                                class_weight='balanced',
                                verbose=1,
                                n_jobs=4)
    # --- Step 4: Estimate RF model and save estimated model:
    rf.fit(X, y)
    rf_dict[num] = rf

# --- Save-off model:
# Specify location and name of object to contain estimated model:
model_object_path = os.path.join(notebook_dir, 'rf.joblib')
# Save estimated model to specified location:
dump(rf_dict, model_object_path)

# Compute OOB error
oob_error_list = [None] * len(n_trees)

# Find OOB error for each forest size: 1500 is the best number
for i in range(len(n_trees)):
    oob_error_list[i] = 1 - rf_dict[n_trees[i]].oob_score_
else:
    # Visualize result:
    fig = plt.figure()
    plt.plot(n_trees, oob_error_list, 'bo',
             n_trees, oob_error_list, 'k')
    fig.suptitle('Error vs Number of trees')
    plt.xlabel('number of trees')
    plt.ylabel('OOB error')


# Feature importance plot
top_num = 10
forest = rf_dict[100]

def plot_feature_imp(num, forest):
    importances = forest.feature_importances_
    # Sort in decreasing order:
    indices = np.argsort(importances)[::-1]
    # Plot the feature importance of the forest
    ax = plt.gca()
    plt.title(f"Top {num} feature importances")
    plt.bar(range(num), importances[indices[0:num]])
    plt.xticks(range(num))
    ax.set_xticklabels(np.array(list(X))[indices[0:num]], rotation=90)
    ax.set_xlabel("Features")
    ax.set_ylabel("Feature Importance")
    plt.show()


plot_feature_imp(top_num, forest)
# --------------------------------------------------
# Part3: Model Validation
# --------------------------------------------------
y_pred_test = forest.predict(X_test)

# Create a confusion matrix to see how accurate model predicts
conf_mat = confusion_matrix(y_true=y_test,
                            y_pred=y_pred_test)

class_names = ['failed', 'successful']
conf_df = pd.DataFrame(conf_mat, class_names, class_names)
# Percentage format for confusion matrix
conf_df_pct = conf_df/conf_df.sum(axis=0)
round(conf_df_pct*100, 1)

# Calculate f1 score of the model which indicates accuracy


def calculate_f1_score(true, pred, average):
    return f1_score(y_true=true,
                    y_pred=pred,
                    average=average)


# Class-level performance:
f1_score_class = calculate_f1_score(y_test, y_pred_test, 'macro')
f1_score_overall = calculate_f1_score(y_test, y_pred_test, 'micro')


# --------------------------------------------------
# Part4: Scoring with user inputs
# --------------------------------------------------

# Replace the input variables with desired inputs for your project
input_row = {'main_category': 'Comics', 'country': 'US',
         'usd_goal_real': 1000, 'project_length': 55}

scoring_input = df_features.iloc[[0]]
scoring_input.loc[:] = 0
to_drop = ['state']
scoring_input.drop(to_drop, inplace=True, axis=1)
scoring_input['usd_goal_real'] = input_row['usd_goal_real']
scoring_input['project_length'] = input_row['project_length']
scoring_input['main_category_' + input_row['main_category']] = 1
scoring_input['country_' + input_row['country']] = 1

scoring_output = forest.predict(scoring_input)
# 1 for successful, 0 for failed
print("is_project_successful: ", scoring_output[0])
