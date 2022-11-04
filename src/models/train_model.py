import click
import sys
import pickle
import json

# pandas
import pandas as pd

#sklearn
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from skopt import BayesSearchCV

from sklearn.svm import SVR

# project
sys.path.append('src')
from data.preprocess import read_processed_data
from models.svr import SVRModel
from data import get_features, get_targets

@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_file):
    print('Training model')

    df = read_processed_data(input_file)
    
    model = SVR()
    
    # define search space
    search_space = dict()
    search_space['C'] = Real(1e-6, 100.0, 'log-uniform')
    search_space['epsilon'] = Real(1e-6, 100.0, 'log-uniform')
    #search_space['degree'] = Integer(1,5)
    search_space['kernel'] = Categorical(['linear', 
                                          #'poly', 
                                          #'rbf', 
                                          #'sigmoid'
                                          ])
    
    kf = KFold(n_splits = 10, shuffle = True, random_state = 42)                                          
    loo = LeaveOneOut() 
    opt = BayesSearchCV(model, 
                     search_space,
                     n_iter = 150,
                     cv = loo,
                     n_jobs = -1,
                     scoring = "neg_mean_absolute_error",
                     random_state = 42, verbose = 3,
                     return_train_score=True
                    )
           
    X = get_features(df)
    y = get_targets(df)
    
    opt.fit(X, y)
    
    
    df = pd.DataFrame(opt.cv_results_)
    df.to_csv(output_file+'.csv')
 
    with open(output_file, 'wb') as ofile:
        pickle.dump(model, ofile, pickle.HIGHEST_PROTOCOL)
   
if __name__ == '__main__':
    main()