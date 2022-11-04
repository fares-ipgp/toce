import click
import sys
sys.path.append('src')

from data.preprocess import read_processed_data
from models.svr import SVRModel

@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))

def main(input_file, output_file):
    print('Training model')

    df = read_processed_data(input_file)
    model = SVRModel()
    model.train(df)
    model.save(output_file)

if __name__ == '__main__':
    main()