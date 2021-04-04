import click
from breed_classifier import Model, DataProvider, logger


@click.group()
def train_cli():
    pass


@train_cli.command()
@click.option("--data-path", required=True, type=str)
@click.option("--save-path", required=True, type=str)
@click.option("--epochs", required=True, type=int)
@click.option("--batch-size", default=1)
@click.option("--gpu/--no-gpu", default=False)
def train(data_path: str, save_path: str, epochs: int, batch_size: int,
          gpu: bool):
    data_provider = DataProvider(root_dir=data_path, batch_size=batch_size)
    model = Model(data_provider=data_provider, use_gpu=gpu,
                  save_path=save_path)
    training_results = model.train(n_epochs=epochs)
    logger.info(f"Training Results: {training_results}")


@click.group()
def test_cli():
    pass


@test_cli.command()
@click.option("--data-path", required=True, type=str)
@click.option("--save-path", required=True, type=str)
@click.option("--gpu/--no-gpu", default=False)
def test(data_path: str, save_path: str, gpu: bool):
    data_provider = DataProvider(root_dir=data_path)
    model = Model(data_provider=data_provider, use_gpu=gpu,
                  save_path=save_path)
    test_results = model.test()
    logger.info(f"Test Results: {test_results}")


main = click.CommandCollection(sources=[train_cli, test_cli])

if __name__ == '__main__':
    main()
