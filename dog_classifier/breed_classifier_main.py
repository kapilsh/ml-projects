import json

import click
import pandas as pd
from breed_classifier import DataProvider, logger
from dog_classifier.breed_classifier import ModelScratch, ModelTransferLearn


@click.group()
def train_cli():
    pass


@train_cli.command()
@click.option("--data-path", required=True, type=str)
@click.option("--save-path", required=True, type=str)
@click.option("--model-results-path", required=True, type=str)
@click.option("--epochs", required=True, type=int)
@click.option("--batch-size", default=1)
@click.option("--gpu/--no-gpu", default=False)
@click.option("--transfer/--no-transfer", default=False)
@click.option("--verbose/--no-verbose", default=False)
def train(data_path: str, save_path: str, model_results_path: str, epochs: int,
          batch_size: int, gpu: bool, transfer: bool, verbose: bool):
    norm_means = [0.485, 0.456, 0.406] if transfer else [0.5, 0.5, 0.5]
    norm_stds = [0.229, 0.224, 0.225] if transfer else [0.5, 0.5, 0.5]
    data_provider = DataProvider(root_dir=data_path, batch_size=batch_size,
                                 norm_means=norm_means, norm_stds=norm_stds)
    clazz = ModelTransferLearn if transfer else ModelScratch
    model = clazz(data_provider=data_provider, use_gpu=gpu, save_path=save_path,
                  verbose=verbose)
    training_results = model.train(n_epochs=epochs)
    logger.info(f"Training Results: {training_results}")
    results_df = pd.DataFrame({"training": training_results.train_losses,
                               "validation": training_results.validation_losses}
                              )
    results_df.to_csv(results_df)


@click.group()
def test_cli():
    pass


@test_cli.command()
@click.option("--data-path", required=True, type=str)
@click.option("--save-path", required=True, type=str)
@click.option("--model-results-path", required=True, type=str)
@click.option("--gpu/--no-gpu", default=False)
@click.option("--transfer/--no-transfer", default=False)
@click.option("--verbose/--no-verbose", default=False)
def test(data_path: str, save_path: str, model_results_path: str, gpu: bool,
         transfer: bool, verbose: bool):
    norm_means = [0.485, 0.456, 0.406] if transfer else [0.5, 0.5, 0.5]
    norm_stds = [0.229, 0.224, 0.225] if transfer else [0.5, 0.5, 0.5]
    data_provider = DataProvider(root_dir=data_path,
                                 norm_means=norm_means, norm_stds=norm_stds)
    clazz = ModelTransferLearn if transfer else ModelScratch
    model = clazz(data_provider=data_provider, use_gpu=gpu, save_path=save_path,
                  verbose=verbose)
    test_results = model.test()
    logger.info(f"Test Results: {test_results}")
    test_results_json = {
        "test_loss": float(test_results.test_loss),
        "correct_labels": int(test_results.correct_labels),
        "total_labels": int(test_results.total_labels),
        "predicted_labels": test_results.predicted_labels.tolist(),
        "target_labels": test_results.target_labels.tolist(),
    }

    with open(model_results_path, 'w') as outfile:
        json.dump(test_results_json, outfile)


main = click.CommandCollection(sources=[train_cli, test_cli])

if __name__ == '__main__':
    main()
