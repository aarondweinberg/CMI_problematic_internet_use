#**An Ordinal (Sequential Binary) Classifier**

#It looks like our attempts so far have under-predicted sii values of 2 and 3. 

#We'll create a class that first predicts whether or not the sii value is 0, then continues upward...
#Note that this isn't quite the same as creating four separate binary predictors for 0, 1, 2, and 3 outcomes.

# There have been numerous ordinal classification algorithms proposed (see, e.g., https://www.sciencedirect.com/science/article/pii/S0888613X16300706)
# We decided to use an algorithm that was proposed by Frank and Hall in 2001 (https://link.springer.com/chapter/10.1007/3-540-44795-4_13)
# (In large part because it was relatively straightforward to use)

from sklearn.base import clone
from sklearn.metrics import accuracy_score
import numpy as np

class OrdinalClassifier():

    def __init__(self, clf, **kwargs):
        self.clf = clf
        self.clf.set_params(**kwargs)
        self.clfs = {}

    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0] - 1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.clf)
                clf.fit(X, binary_y)
                #print('binary_y has been fit for', self.unique_class[i])
                self.clfs[i] = clf

    def predict_proba(self, X):
        clfs_predict = {k: v.predict_proba(X) for k, v in self.clfs.items()}
        predicted = []
        for i, y in enumerate(self.unique_class):
            #print('encoding for i=', i)
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[i][:, 1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                predicted.append(clfs_predict[i - 1][:, 1] - clfs_predict[i][:, 1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[i - 1][:, 1])
        return np.vstack(predicted).T

    def predict(self, X):
        return self.unique_class[np.argmax(self.predict_proba(X), axis=1)]
    
    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)