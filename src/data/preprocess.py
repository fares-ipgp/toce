import click
import pandas as pd

def get_columns_features():
    return [ #'depth_m',
             #'facies_num',
             #'bioturb',
             #'amb1',
             #'amb2',
             'drx_qz',
             'drx_pl',
             'drx_fk',
             'drx_ca',
             'drx_py',
             'drx_arc',
             'drx_arc_i',
             'drx_arc_is',
             'drx_arc_cl',
             'drx_arc_k']

def get_columns_targets():
    return 'toc'

def get_features(df):
    '''returns features'''
    columns_features = get_columns_features()
    return df[columns_features]

def get_targets(df):
    '''returns targets'''
    columns_targets = get_columns_targets()
    return df[columns_targets]

def read_raw_data(fname='data/raw/toce.csv'):
    df = pd.read_csv(fname)
    return df

def preprocess_data(df):
    df = df.copy().dropna()  # I want to avoid inplace modifications
    return df

def read_processed_data(fn='data/processed/processed.pickle'):
    df = pd.read_pickle(fn)
    return df

@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_file):
    print('Preprocessing data')

    # read
    df = read_raw_data(input_file)
    
    # preprocess
    df = preprocess_data(df)

    # write
    df.to_pickle(output_file)

if __name__ == '__main__':
    main()