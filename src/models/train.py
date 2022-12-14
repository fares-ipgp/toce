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
from mlflow.models.signature import infer_signature

#sklearn
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
@click.argument('model', type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_file, model):
    
    mlflow.set_experiment(model)
    
    run_params = {}
    run_params['model']=model
    
    print('Training model')

    df = read_processed_data(input_file)

    pre = StandardScaler()
    svr = SVR()    
    model = Pipeline(steps=[('pre', pre), ('svr', svr) ])      

    
    # define search space
    search_space = dict()
    search_space['svr__C'] = Real(1e-6, 100.0, 'log-uniform')
    search_space['svr__epsilon'] = Real(1e-6, 100.0, 'log-uniform')
    search_space['svr__kernel'] = Categorical(['linear', 
                                          #'poly', 
                                          'rbf' 
                                          #'sigmoid'
                                          ])
    # cross validation
    kf = KFold(n_splits = 20, shuffle = True, random_state = 42)                                          
    loo = LeaveOneOut() 
    
    # Hyperparameter search
    opt = BayesSearchCV(model, 
                     search_space,
                     n_iter = 5,
                     cv = loo,
                     n_jobs = -1,
                     scoring = "neg_mean_absolute_error",
                     random_state = 42, verbose = 3,
                     return_train_score=True,
                     refit=True
                    )

    # prepare data for training
    X = get_features(df)
    X.iloc[:] = X.iloc[:].astype(float)
    y = get_targets(df)

    # start run
    mlflow.start_run()

    # fit model    
    opt.fit(X, y)

    # get metrics    
    log_metric = ['mean_test_score',
                  'std_test_score',
                  'mean_train_score',
                  'std_train_score'
                 ]
    df_metrics = pd.DataFrame(opt.cv_results_).sort_values(by='rank_test_score', ascending=False)[log_metric]
    df_metrics.to_csv(output_file+'_metrics.csv')
    metrics = df_metrics.to_dict(orient="records")
    
    # log run params        
    mlflow.log_params(run_params)

    # log optimizer params
    ic(opt.get_params())        
    mlflow.log_params(opt.get_params())
    
    # log best model params        
    mlflow.log_params(opt.best_params_)
    
    # log model 
    signature = infer_signature(X, y)
    mlflow.sklearn.log_model(opt.best_estimator_, 'model',serialization_format='pickle',signature =signature)
        
    # log metrics        
    for step,metric in enumerate(metrics):
        mlflow.log_metrics(metric,step)
       
    # save model
    with open(output_file, 'wb') as ofile:
        pickle.dump(opt.best_estimator_, ofile, pickle.HIGHEST_PROTOCOL)

    # log artifacts    
    mlflow.log_artifact(input_file)
    mlflow.log_artifact(output_file)

    # end run
    mlflow.end_run()
    
if __name__ == '__main__':
    main()