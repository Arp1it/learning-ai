# Train a logistic regression classifier to predict whether a flower is iris virginica or not

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()

# print(list(iris.keys()))
# print(iris['data'])
# print(iris['data'].shape)
# print(iris['target'])
# print(iris['DESCR'])

# X = iris["data"][:, 3]
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int_)
# print(y)
# print(X)

# Train a logistic Classifier
clf = LogisticRegression()
clf.fit(X,y)
exaple = clf.predict([[2.6]])
print(exaple)

# Using matplotlib to plot the visualization
X_new = np.linspace(0,3,1000).reshape(-1, 1)
print(X_new)
Y_prob = clf.predict_proba(X_new)
print(Y_prob)
plt.plot(X_new, Y_prob[:, 1], "g-", label="virginica")
plt.show()