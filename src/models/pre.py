import pickle
import sys

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# project
sys.path.append('src')
from models.base import BaseModel

# SVR model
class Preprocess(BaseModel):
    
    def __init__(self):

        # Model name              
        self.name = 'Preprocessor'
                
        # Standard scaler
        num = StandardScaler()
        
        # Model
        self.model = num

