import click
import pandas as pd
import json

from loguru import logger


def calculate_cardinality_and_tokenizer_values(column):
    cardinality = column.nunique()
    tokenizer_values = sorted(column.unique().tolist())
    return cardinality, tokenizer_values


def calculate_mean_and_std(column):
    mean = float(column.mean())
    std = float(column.std())
    return mean, std


@click.command()
@click.option('--file_path',
              type=click.Path(exists=True),
              help='Path to the parquet file',
              required=True)
@click.option('--output_path',
              type=click.Path(),
              default='output.json',
              help='Path to the output JSON file')
def process_file(file_path, output_path):
    """
    Process the file specified by --file_path.
    """
    logger.info("Reading the parquet file {}...".format(file_path))
    df = pd.read_parquet(file_path)
    results = {}
    for column_name in df.columns:
        if column_name.startswith("SPARSE"):
            logger.info("Processing column: {}".format(column_name))
            cardinality, tokenizer_values = calculate_cardinality_and_tokenizer_values(df[column_name])
            results[column_name] = {
                'cardinality': cardinality,
                'tokenizer_values': tokenizer_values
            }
        elif column_name.startswith("DENSE"):
            logger.info("Processing column: {}".format(column_name))
            mean, std = calculate_mean_and_std(df[column_name])
            results[column_name] = {
                'mean': mean,
                'std': std
            }
    with open(output_path, 'w') as f:
        logger.info("Writing results to {}".format(output_path))
        json.dump(results, f)


if __name__ == "__main__":
    process_file()
