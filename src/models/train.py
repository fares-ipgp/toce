import click
import sys
import pickle
import json

from icecream import ic

# pandas
import pandas as pd

# mflow
import mlflow
import mlflow.sklearn

#sklearn
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor

#metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


# skopt
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# project
sys.path.append('src')
from data.preprocess import read_processed_data
from models.svr import SVRModel
from data import get_features, get_targets

def eval_metrics(y, y_pred):
    metrics = {}
    metrics['r2']=r2_score(y,y_pred)        
    metrics['mse']=mean_squared_error(y,y_pred)
    metrics['mae']=mean_absolute_error(y,y_pred)
    metrics['mape']=mean_absolute_percentage_error(y,y_pred)
    return metrics

def log_results(res):
    print(res)

@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_file):
    
    mlflow.set_experiment('toce-mayer')
    
    #mlflow.sklearn.autolog()
    
    print('Training model')

    df = read_processed_data(input_file)
    
          
    model = LinearSVR()
    
    # define search space
    search_space = dict()
    search_space['C'] = Real(1e-6, 100.0, 'log-uniform')
    search_space['epsilon'] = Real(1e-6, 100.0, 'log-uniform')
    search_space['loss'] = Categorical(['epsilon_insensitive', 
                                          'squared_epsilon_insensitive'])
   
    # optimizers
    kf = KFold(n_splits = 10, shuffle = True, random_state = 42)                                          
    loo = LeaveOneOut() 
    
    opt = BayesSearchCV(model, 
                     search_space,
                     n_iter = 40,
                     cv = loo,
                     n_jobs = -1,
                     scoring = "neg_mean_absolute_error",
                     random_state = 42, verbose = 3,
                     return_train_score=True
                    )
           
    X = get_features(df)
    y = get_targets(df)
    
    opt.fit(X, y)
 
    log_params = [ 'param_C',
                  'param_epsilon',
                  'param_loss'
                 ]

    log_metric = ['mean_test_score',
                  'std_test_score',
                  'rank_test_score',
                  'mean_train_score',
                  'std_train_score',
                  'rank_train_score'
                 ]
    
    
    df_params = pd.DataFrame(opt.cv_results_).sort_values(by='rank_test_score', ascending=False)[log_params]
    df_params.to_csv(output_file+'_params.csv')

    df_metrics = pd.DataFrame(opt.cv_results_).sort_values(by='rank_test_score', ascending=False)[log_metric]
    df_metrics.to_csv(output_file+'_metrics.csv')
    
    params = df_params.head(1).to_dict(orient="records")
    ic(params)

    
    metrics = df_metrics.to_dict(orient="records")
    ic(metrics)

    
    mlflow.start_run()
    for step,param in enumerate(params):
        mlflow.log_params(opt.best_params_)
        
    for step,metric in enumerate(metrics):
        mlflow.log_metrics(metric,step)
    mlflow.end_run()
    

    #with open(output_file, 'wb') as ofile:
    #    pickle.dump(model, ofile, pickle.HIGHEST_PROTOCOL)
   
if __name__ == '__main__':
    main()