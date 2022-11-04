import pickle
import sys

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

# project
sys.path.append('src')
from models.base import BaseModel
from models.pre import Preprocess

# SVR model
class SVRModel(BaseModel):
    
    def __init__(self):
        
        # Model name
        self.name = 'Support Vector Regression'
        
        pre = Preprocess()
        svr = SVR(C=1, epsilon=0.01, kernel='rbf',verbose=1)
               
        # Model 
        self.model = Pipeline(steps=[('pre', pre.get_model()), ('svr', svr) ])
        
