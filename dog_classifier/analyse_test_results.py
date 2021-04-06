import json
import numpy as np
import pandas as pd


def read_json_file(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)


#%%

if __name__ == '__main__':
    results_file = "/home/ksharma/tmp/test_results_transfer.json"
    results = read_json_file(results_file)

    labels_df = pd.DataFrame(
        {"predicted": np.array(results["predicted_labels"]).astype(int),
         "actual": np.array(results["target_labels"]).astype(int)})

    print(labels_df)