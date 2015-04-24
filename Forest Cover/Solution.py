# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
import matplotlib as plt

# <codecell>

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sampleSubmission.csv')

# <codecell>

train_data.head()

# <codecell>

train_data.describe()

# <codecell>

train_data.dtypes

# <codecell>

train_data.Cover_Type = pd.Categorical(train_data.Cover_Type)

# <codecell>

train_data.dtypes

# <codecell>

train_indep = train_data.drop(['Id', 'Cover_Type'], axis = 1)
train_dep = train_data.Cover_Type

test_X = test_data.drop(['Id'], axis = 1)

# <codecell>

train_X, val_X, train_y, val_y = train_test_split(train_indep,
                                                  train_dep, test_size = 0.25, random_state = 42)

# <codecell>

train_X.shape

# <codecell>

train_y.shape

# <codecell>

val_X.shape

# <codecell>

val_y.shape

# <codecell>

# One-vs-All Logistic Regression
parameters = {'C': [0.001, 0.01, 0.1, 1, 3, 10, 100]}
logit_mod = LogisticRegression()
clf = GridSearchCV(logit_mod, parameters, verbose = 1, n_jobs = -1)

# <codecell>

fit = clf.fit(train_X, train_y)

# <codecell>

logit_mod = LogisticRegression(C = 10)
fit = logit_mod.fit(train_X, train_y)

# <codecell>

predict = fit.predict(val_X)

# <codecell>

predict

# <codecell>

score = np.sum(predict == val_y)/float(len(predict))

# <codecell>

score

# <codecell>

# This doesn't seem to be like a very good score
# Let us try linearSVC
from sklearn.svm import LinearSVC

# <codecell>

parameters = {'C': [0.001, 0.01, 0.1, 1, 3, 10, 30, 100]}
linear_svc = LinearSVC()
clf = GridSearchCV(linear_svc, parameters, verbose = 1, n_jobs = 3)

# <codecell>

fit = clf.fit(train_X, train_y)

# <codecell>

fit.best_params_

# <codecell>

linear_svc = LinearSVC(C = 10)
fit = linear_svc.fit(train_X, train_y)
predict = fit.predict(val_X)
score = np.sum(predict == val_y)/float(len(predict))
score

# <codecell>

# This is even worse
# Finally let's use a random Forest model
from sklearn.ensemble import RandomForestClassifier as rf

# <codecell>

parameters = {'n_estimators': [1000, 2000, 5000, 10000]}
rf_classif = rf()
clf = GridSearchCV(rf_classif, parameters, verbose = 1, n_jobs = 4)
fit = clf.fit(train_X, train_y)
fit.best_params_

# <codecell>

rf_classif = rf(n_estimators = 2000, verbose = 1, n_jobs = 4, max_features = 15)
fit = rf_classif.fit(train_X, train_y)

# <codecell>

predict = fit.predict(val_X)

# <codecell>

score = np.sum(predict == val_y)/float(len(predict))
score

# <codecell>

predict = fit.predict(test_X)

# <codecell>

sample_submission.head()

# <codecell>

result = pd.DataFrame(test_data.Id, columns=['Id'])

# <codecell>

result['Cover_Type'] = predict

# <codecell>

result.head()

# <codecell>

result.to_csv('first_rf.csv', index=False)

# <codecell>


