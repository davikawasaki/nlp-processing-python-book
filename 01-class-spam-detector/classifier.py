from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np

# Loading file and turning CSV into matrix
data = pd.read_csv('spambase.data').as_matrix()
np.random.shuffle(data)

# Splice and get all rows until column 48
X = data[:, :48]
# Splice, reverse and get last column (true label)
Y = data[:, -1]

# Get all rows except last 100
Xtrain = X[:-100, ]
Ytrain = Y[:-100, ]

# Get last 100 rows as testing
Xtest  = X[-100:, ]
Ytest  = Y[-100:, ]

# Classifying with NaiveBayes
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print "Classification rate for NB: ", model.score(Xtest, Ytest)

# Classifying with Ensemble Based
model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print "Classification rate for AdaBoost: ", model.score(Xtest, Ytest)
