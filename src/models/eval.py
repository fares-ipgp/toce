# sklearn
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

def eval_reg(model,X,y):

    #predict    
    y_pred = model.predict(X)
    
    #calculate metrics
    metrics = {}
    metrics['r2']=r2_score(y,y_pred)        
    metrics['mse']=mean_squared_error(y,y_pred)
    metrics['mae']=mean_absolute_error(y,y_pred)
    metrics['mape']=mean_absolute_percentage_error(y,y_pred)
    
    return metrics

def eval_clf(model,X,y):

    #predict    
    y_pred = model.predict(X)
    
    #calculate metrics
    metrics = {}
    metrics['r2']=r2_score(y,y_pred)        
    metrics['mse']=mean_squared_error(y,y_pred)
    metrics['mae']=mean_absolute_error(y,y_pred)
    metrics['mape']=mean_absolute_percentage_error(y,y_pred)
    
    return metrics
