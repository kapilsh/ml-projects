import pandas as pd
import numpy as np
import click

NUM_DENSE = 13
NUM_SPARSE = 26

dense_features = [f"DENSE_{i}" for i in range(NUM_DENSE)]
sparse_features = [f"SPARSE_{i}" for i in range(NUM_SPARSE)]

columns = ["labels"] + dense_features + sparse_features


# Function to convert hexadecimal string to integer
def hex_to_int(hex_string):
    if hex_string == 'nan':
        return -1
    return int(hex_string, 16)


def clean_chunk(chunk):
    chunk[dense_features] = chunk[dense_features].fillna(0).astype(np.float32)
    for sparse_feature_name in sparse_features:
        chunk[sparse_feature_name] = chunk[sparse_feature_name].astype(
            str).apply(hex_to_int)
    chunk["labels"] = chunk["labels"].astype(np.float64)
    return chunk


def save_smaller_data_to_parquet(file_path: str) -> None:
    chunk_size = 1000000
    all_dfs = []
    num_chunks = 10
    original_chunks = 10

    for chunk in pd.read_csv(file_path, sep='\t', header=None, names=columns,
                             compression='gzip', chunksize=chunk_size):
        print(f"Done with {original_chunks - num_chunks}")
        chunk = clean_chunk(chunk)
        all_dfs.append(chunk)
        num_chunks -= 1
        if num_chunks == 0:
            break
    result_df = pd.concat(all_dfs)
    # Display the resulting DataFrame
    print("Total number of rows loaded", len(result_df))
    print("Total data size", result_df.memory_usage(deep=True).sum() / 10 ** 9,
          "gb")

    save_file_path = file_path.replace(".", "_") + "_converted.parquet"
    result_df.to_parquet(save_file_path)


@click.command()
@click.option('--file_path', type=click.Path(exists=True),
              help='Path to the file to process')
def main(file_path: str):
    """
    Process the file specified by FILE_PATH.
    """
    click.echo(f"Processing file: {file_path}")
    save_smaller_data_to_parquet(file_path=file_path)


if __name__ == "__main__":
    main()
