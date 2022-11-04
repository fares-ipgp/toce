import pickle
import sys

# sklearn
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

# project
sys.path.append('src')
from data import get_features, get_targets

class BaseModel(object):

    def __init__(self):
       
        # Model name
        self.name = 'Base Model'
        
        # Model
        self.model = BaseEstimator()

    def get_model(self):
        return self.model
    
    def get_params(self):
        return self.model.get_params()

    def train(self, df):
        X = get_features(df)
        y = get_targets(df)
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred
    
    def eval(self, df):
        X = get_features(df)
        y = get_targets(df)
        
        y_pred = self.predict(X)
        
        metrics = {}
        metrics['r2']=r2_score(y,y_pred)        
        metrics['mse']=mean_squared_error(y,y_pred)
        metrics['mae']=mean_absolute_error(y,y_pred)
        metrics['mape']=mean_absolute_percentage_error(y,y_pred)
        
        return metrics
        

    def save(self, fn):
        with open(fn, 'wb') as ofile:
            pickle.dump(self.model, ofile, pickle.HIGHEST_PROTOCOL)

    def load(self, fn):
        with open(fn, 'rb') as ifile:
            self.model = pickle.load(ifile)
