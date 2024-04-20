import click
import pandas as pd
import json

from loguru import logger


def calculate_metadata(column):
    cardinality = column.nunique()
    tokenizer_values = dict(
        zip(column.unique().tolist(), list(range(cardinality))))
    return cardinality, tokenizer_values


@click.command()
@click.option('--file_path',
              type=click.Path(exists=True),
              help='Path to the parquet file')
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
            cardinality, tokenizer_values = calculate_metadata(df[column_name])
            results[column_name] = {
                'cardinality': cardinality,
                'tokenizer_values': tokenizer_values
            }
    with open(output_path, 'w') as f:
        logger.info("Writing results to {}".format(output_path))
        json.dump(results, f)


if __name__ == "__main__":
    process_file()
