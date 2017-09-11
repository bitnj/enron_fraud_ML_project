#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from pprint import pprint

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
        'loan_advances', 'bonus', 'restricted_stock_deferred', 
        'deferred_income', 'total_stock_value', 'expenses',
        'exercised_stock_options', 'other', 'long_term_incentive',
        'restricted_stock', 'director_fees', 'to_messages', 
        'from_poi_to_this_person', 'from_messages',
        'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


# *****************************************************************************
# Task 3: Create new feature(s)
#
# Pandas code for translating dictionary to pandas and back to dictionary is
# from forum post:
# https://discussions.udacity.com/t/pickling-pandas-df/174753/2
# *****************************************************************************
import pandas as pd
from pandas.tools.plotting import scatter_matrix

df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))

# set the index of the df to be the employees series
df.set_index(employees, inplace=True)

# convert string representation of NaN to actual NaN
df.replace(to_replace='NaN', value=np.nan, inplace=True)

# gather some descriptive stats on the dataframe
# number of records
num_records = df.shape[0]
num_features = len(df.columns)
num_poi = df['poi'].sum()

# % of each column that is NaN
pct_nan = df.isnull().sum() / num_records

# % of incoming emails from a POI
pct_from_poi = df['from_poi_to_this_person'] / df['to_messages']

# % of outgoing emails to a POI
pct_to_poi = df['from_this_person_to_poi'] / df['from_messages']

# ratio of bonus to salary
bonus_to_salary_ratio = df['bonus'] / df['salary']

# gap between bonus and salary
bonus_salary_gap = df['bonus'] - df['salary']

# capture some information for answering the project questions
#with open("df_stats.txt", "w") as text_file:
#    text_file.write("Number of records: {}\n\n".format(num_records))
#    text_file.write("Number of features: {}\n\n".format(num_features))
#    text_file.write("Number of POIs: {}\n\n".format(num_poi))
#    text_file.write("POI List:\n {}\n\n".format(df.loc[df['poi']==1]['poi']))
#    text_file.write("Pct NaN:\n {}\n\n".format(pct_nan))
#    text_file.write("Pct emails from POI:\n {}\n\n".format(pct_from_poi))
#    text_file.write("Pct emails to POI:\n {}\n\n".format(pct_to_poi))
            
# get an overview of the variables
#scatter_matrix(df, alpha=0.2, figsize=(6,6), diagonal='kde')

#x = df['salary'] + df['bonus']
#y = pct_to_poi
#plt.scatter(x, y, c=df['poi'], alpha=0.9)
#plt.show()

#x = bonus_to_salary_ratio
#y = pct_to_poi
#plt.scatter(x, y, c=df['poi'], alpha=0.9)
#plt.show()

# add new feature(s) to the dataframe
add_new_features = True
if add_new_features:
    #df['bonus_to_salary_ratio'] = pd.Series(bonus_to_salary_ratio, index=df.index) 
    #df['pct_email_from_poi'] = pd.Series(pct_from_poi, index=df.index)
    df['bonus_salary_gap'] = pd.Series(bonus_salary_gap, index=df.index)
    df['pct_email_to_poi'] = pd.Series(pct_to_poi, index=df.index)

# replace NaN with 0
df.fillna(value=0, inplace=True)

# get pairwise correlations for all features
df_corr = df.corr()
df_corr.to_csv(r'correlations.csv', header=True, columns=features_list,
        index=True, sep=',', mode='w')

# new features list after creating / removing / modifying
cols_to_drop = ['email_address']
df = df.drop(cols_to_drop, axis=1)
new_features_list = df.columns.values
new_features_list = new_features_list.tolist()
print(new_features_list)

# put POI back in the first position in the list
old_index = new_features_list.index('poi')
new_features_list.insert(0, new_features_list.pop(old_index))

# create a dictionary from the dataframe
df_dict = df.to_dict('index')

# Store to my_dataset for easy export below.
my_dataset = df_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)


# *****************************************************************************
# Set up to allow testing of models similar to the tester.py used by the grader
# and then test out different classification algorithms
# *****************************************************************************
from sklearn.model_selection import GridSearchCV # parameter tuning
from sklearn.pipeline import Pipeline
from sklearn import preprocessing       # scale
from sklearn.decomposition import PCA   # reduce
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedShuffleSplit # given the small
# number of POIs in the dataset we need to fit our model using reasonably large
# number of training sets 

cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
scaler = preprocessing.StandardScaler()
selector = SelectKBest()
pca = PCA()

#features_train_scaled = scaler.fit_transform(features_train)
#features_test_scaled = scaler.transform(features_test)
#selector.fit(features_train_scaled, labels_train)
#is_selected = selector.get_support()
#weights = selector.scores_

#for i in range(0, len(weights)):
#    if is_selected[i]:
#        print(new_features_list[i + 1],':' , weights[i])

# pca
#features_train_pca = pca.fit_transform(features_train_scaled)
#features_test_pca = pca.transform(features_test_scaled)
#print(pca.components_)


# *****************************************************************************
# Naive Bayes
# *****************************************************************************
from sklearn.naive_bayes import GaussianNB

clf_NB = GaussianNB()

# Set up a pipeline to do all the work - scale, reduce, select, classify
#pipe_NB = Pipeline([
#    ('scale', scaler),
#    ('select', selector),
#    ('reduce', pca),
#    ('classify', clf_NB)])

# a dictionary of parameters for GridSearchCV
#param_grid_NB = {
#        'select__k': range(1, len(new_features_list)),
#        'reduce__n_components': [0.90, 0.925, 0.95, 0.975],
#        'reduce__whiten': [False, True]
        }

#clf_NB = GridSearchCV(pipe_NB, param_grid_NB, cv = cv, scoring = 'f1')
#clf_NB.fit(features, labels)

#features_selected_bool = clf_NB.best_estimator_.named_steps['select'].get_support()
#features_selected_list = [x for x, y in zip(new_features_list[1:], features_selected_bool) if y]
#features_selected_scores = [x for x, y in
#        zip(clf_NB.best_estimator_.named_steps['select'].scores_,
#            features_selected_bool) if y]
#print(features_selected_list)
#print(features_selected_scores)
#print(clf_NB.best_params_)


# *****************************************************************************
# Decision Tree
# *****************************************************************************
from sklearn import tree

clf_DT = tree.DecisionTreeClassifier()

# Set up a pipeline to do all the work - scale, select, classify
#pipe_DT = Pipeline([
#    ('scale', scaler),
#    ('select', selector),
#    ('classify', clf_DT)])

# a dictionary of parameters for GridSearchCV
#param_grid_DT = {
#        'select__k': range(1, len(new_features_list)),
#        'classify__criterion': ['gini', 'entropy']
        }

#clf_DT = GridSearchCV(pipe_DT, param_grid_DT, cv = cv, scoring = 'f1')
#clf_DT.fit(features, labels)

#features_selected_bool = clf_DT.best_estimator_.named_steps['select'].get_support()
#features_selected_list = [x for x, y in zip(new_features_list[1:], features_selected_bool) if y]
#features_selected_scores = [x for x, y in
#        zip(clf_DT.best_estimator_.named_steps['select'].scores_,
#            features_selected_bool) if y]
#print(features_selected_list)
#print(features_selected_scores)
#print(clf_DT.best_params_)


# *****************************************************************************
# SVM
# *****************************************************************************
from sklearn.svm import SVC

clf_SVM = SVC()

# set up a pipleline to do all the work - scale, select, classify
#pipe_SVM = Pipeline([
#    ('scale', scaler),
#    ('select', selector),
#    ('classify', clf_SVM)])

# a dictionary of parameters for GridSearchCV
#param_grid_SVM = {
#        'select__k': range(1, len(new_features_list)),
#        'classify__C': [1, 100, 1000]
#        }

#clf_SVM = GridSearchCV(pipe_SVM, param_grid_SVM, cv = cv, scoring = 'f1')
#clf_SVM.fit(features, labels)

#features_selected_bool = clf_SVM.best_estimator_.named_steps['select'].get_support()
#features_selected_list = [x for x, y in zip(new_features_list[1:], features_selected_bool) if y]
#features_selected_scores = [x for x, y in
#        zip(clf_SVM.best_estimator_.named_steps['select'].scores_,
#            features_selected_bool) if y]
#print(features_selected_list)
#print(features_selected_scores)
#print(clf_SVM.best_params_)


# *****************************************************************************
# K Nearest Neighbors
# *****************************************************************************
from sklearn.neighbors import KNeighborsClassifier

clf_KNN = KNeighborsClassifier()

# set up a pipeline to do all the work - scale, select, classify
pipe_KNN = Pipeline([
#    ('scale', scaler),
    ('select', selector),
    ('classify', clf_KNN)])

param_grid_KNN = {
        'select__k': range(1, len(new_features_list)),
        'classify__algorithm': ['ball_tree', 'kd_tree'],
        'classify__n_neighbors': range(2, 6),
        'classify__weights': ['uniform', 'distance']
        }

clf_KNN = GridSearchCV(pipe_KNN, param_grid_KNN, cv = cv, scoring = 'f1')
clf_KNN.fit(features, labels)

features_selected_bool = clf_KNN.best_estimator_.named_steps['select'].get_support()
features_selected_list = [x for x, y in zip(new_features_list[1:], features_selected_bool) if y]
features_selected_scores = [x for x, y in
        zip(clf_KNN.best_estimator_.named_steps['select'].scores_,
            features_selected_bool) if y]
print(features_selected_list)
print(features_selected_scores)
print(clf_KNN.best_params_)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf_KNN.best_estimator_, my_dataset, new_features_list)
