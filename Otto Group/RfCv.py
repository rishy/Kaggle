# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from ggplot import *
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as rfClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import balance_weights
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

#rf = GridSearchCV(rfClassifier(), [{'n_estimators': [10, 50, 100, 150, 200, 300, 500]}, 
                                   #{'max_features': ["sqrt", "log2", None]}])
rf = rfClassifier(n_estimators = 500, max_features = 18, verbose = 1)

# <codecell>

rf_fit = rf.fit(train_X, train_y, sample_weight = balance_weights(train_y))

# <codecell>

rf_fit

# <codecell>

rf_prob = rf_fit.score(val_X, val_y)

# <codecell>

rf_prob

# <codecell>

rf_prob = rf_fit.predict_proba(test_X)

# <codecell>

df = DataFrame(rf_prob, columns = np.unique(train_y))
df.insert(loc = 0, column = "id", value = test["id"])

# <codecell>

df.to_csv("RfWeightedSolution.csv", index = False)

# <codecell>


