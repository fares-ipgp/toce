import pickle
import sys

# sklearn
from sklearn.svm import SVR

# project
sys.path.append('src')
from models.base import BaseModel

# SVR model
class SVRModel(BaseModel):
    
    def __init__(self):
        
        # Model name
        self.name = 'Support Vector Regression'
        
        # Model 
        self.model = SVR(C=1, epsilon=0.01, kernel='rbf')
        


