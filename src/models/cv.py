#sklearn
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

def build_cv_loo():

    # leave one out
    loo = LeaveOneOut() 
    return loo

def build_cv_kfold(n_splits = 20, random_state = 42):
    
    # kfold
    kf = KFold(n_splits = n_splits, 
               shuffle = True, 
               random_state = 42)                                          
    return kf