# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from ggplot import *
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier as rfClassifier
import pandas as pd
from pandas import DataFrame

# <codecell>

# Read train and test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Final training dataframe after dropping "id" and "target" cols
train.X = train.drop(["id", "target"], axis = 1)
train.y = train["target"]

test.X = test.drop(["id"], axis = 1)

# <codecell>

# Train head
train.head()

# <codecell>

# Basic Column Statistics
train.describe()

# <codecell>

# PCA for visualizations
pca = PCA(n_components=2)

# <codecell>

pca.fit(train.X)

# <codecell>

df_pca = DataFrame(pca.transform(train.X), columns = ["x", "y"])
df_pca["target"] = train.y

# <codecell>

# Using ggplot to get a scatter plot
qplot('x', 'y', data = df_pca, color = 'target') 

# <codecell>

# Histogram of target classes
qplot(train.y)

# <codecell>

linearSvc = GridSearchCV(LinearSVC(), [{'C': [1]}])
#linearSvc = LinearSVC(C = 100)
svcFit = linearSvc.fit(train.X, train.y)

# <codecell>

svcPredict = svcFit.predict(train.X)

# <codecell>

svcFit.score(train.X, train.y)

# <codecell>

np.shape(train.y)

# <codecell>

write.csv()

# <codecell>

rf = rfClassifier(n_estimators = 500)

# <codecell>

rfFit = rf.fit(train.X, train.y)

# <codecell>

rfFit.score(train.X, train.y)

# <codecell>

rf.prob = rfFit.predict_proba(test.X)

# <codecell>

df = DataFrame(rf.prob, columns = np.unique(train.y))
df.insert(loc = 0, column = "id", value = test["id"])

# <codecell>

df.head()

# <codecell>

df.to_csv("simpleRf.csv", )

