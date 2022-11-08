# skopt
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

def build_ss_bs():
    
   # define search space
    ss = dict()
    ss['svr__C'] = Real(1e-6, 100.0, 'log-uniform')
    ss['svr__epsilon'] = Real(1e-6, 100.0, 'log-uniform')
    ss['svr__kernel'] = Categorical(['linear', 'rbf'])
    return ss

def build_opt_bs(model,search_space,cv,n_iter = 5,scoring = "neg_mean_squared_error",random_state = 42, verbose = 3,):
    
    opt = BayesSearchCV(model, 
                        search_space,
                        n_iter = n_iter,
                        cv = cv,
                        n_jobs = -1,
                        scoring = scoring,
                        random_state = random_state, 
                        verbose = verbose,
                        return_train_score=True,
                        refit=True
    )
    return opt
