
import click
import matplotlib

matplotlib.use('agg')
import seaborn as sns

import sys

sys.path.append('src')

from data.preprocess import read_processed_data
from data.preprocess import get_columns_features, get_columns_targets 

def pairplot(df):
    g = sns.PairGrid(df,vars = get_columns_features(), hue=get_columns_targets(), diag_sharey=False, corner=True)
    g.map_lower(sns.scatterplot)
    g.map_diag(sns.histplot,fill=False , hue=None, element="poly")
    g.add_legend()
    return g

def exploratory_visualization(df):
    return pairplot(df)

@click.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_file):
    print('Plotting pairwise distribution...')
    df = read_processed_data(input_file)
    plot = exploratory_visualization(df)
    plot.savefig(output_file)

if __name__ == '__main__':
    main()