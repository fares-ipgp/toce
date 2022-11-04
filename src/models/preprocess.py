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
class PreProcess(BaseModel):
    
    def __init__(self):

        # Model name              
        self.name = 'Preprocessor'
                
        # Column transformer (Only Numeric)
        num = StandardScaler()
        cols = ColumnTransformer( transformers = [('num', num)] )
        
        # Model
        self.model = Pipeline(steps=[('cols',cols)]

