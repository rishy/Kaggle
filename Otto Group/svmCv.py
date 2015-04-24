# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from ggplot import *
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as rfClassifier
from sklearn.cross_validation import train_test_split
import pandas as pd
from pandas import DataFrame

# <codecell>

# Read train and test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Final training dataframe after dropping "id" and "target" cols
data_X = train.drop(["id", "target"], axis = 1)
data_y = train["target"]

train_X, val_X, train_y, val_y = train_test_split(data_X, data_y,
                                                  test_size = 0.33, random_state = 42)

test_X = test.drop(["id"], axis = 1)

# <codecell>

tuning_params = [{'C': [0.01, 0.1, 1, 10, 50, 100]},
                 {'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}]
svm = GridSearchCV(SVC(probability = True), param_grid = tuning_params)

# <codecell>

svm_fit = svm.fit(train_X, train_y)

# <codecell>


