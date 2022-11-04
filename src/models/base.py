import pickle
import sys

# sklearn
from sklearn.base import BaseEstimator

# project
sys.path.append('src')
from data import get_features, get_targets

class BaseModel(object):

    def __init__(self):
       
        # Model name
        self.name = 'Base Model'
        
        # Model
        self.model = BaseEstimator()

    def get_params(self):
        return self.model.get_params()

    def train(self, df):
        X = get_features(df)
        y = get_targets(df)
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred

    def save(self, fn):
        with open(fn, 'wb') as ofile:
            pickle.dump(self.model, ofile, pickle.HIGHEST_PROTOCOL)

    def load(self, fn):
        with open(fn, 'rb') as ifile:
            self.model = pickle.load(ifile)
