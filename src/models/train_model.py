import click
import sys

# pandas
import pandas as pd

#sklearn
from sklearn.model_selection import GridSearchCV

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
    model = SVRModel()
    
    #model.train(df)
    
    # Metrics
    scoring = ['neg_mean_absolute_error']
    params = [{
            'svr__C': [0.01,0.1,1,2,3,4,5,6],
            'svr__epsilon': [0.001,0.005,0.01,0.01,0.1,1],
            'svr__kernel': ['rbf']
          }]
  
    gs = GridSearchCV(model.model,
                param_grid=params,
                scoring=scoring,
                refit='neg_mean_absolute_error',
                cv=20, 
                return_train_score=True,
                verbose=3)
    
    X = get_features(df)
    y = get_targets(df)
    gs.fit(X,y)
    
    result_cols = ['mean_fit_time',
                'mean_score_time' ,
                    'params',
                'mean_test_neg_mean_absolute_error',
                'std_test_neg_mean_absolute_error',
                'rank_test_neg_mean_absolute_error',
                'mean_train_neg_mean_absolute_error',
                'std_train_neg_mean_absolute_error']
      
    df_results = pd.DataFrame.from_dict(gs.cv_results_, orient='columns')
    df_results = df_results[result_cols].sort_values(by='rank_test_neg_mean_absolute_error')
    #print('\nMAE: ', model.eval(df))
    
    df_results.to_csv('metrics.csv',sep="\t")
    
    model.save(output_file)

if __name__ == '__main__':
    main()