import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from loguru import logger
import json

from tqdm import tqdm

from criteo_dataset import CriteoParquetDataset
from model import DLRM, read_metadata, Parameters as ModelParameters


def main():
    # Load hyperparameters
    with open('model_hyperparameters.json', 'r') as f:
        hyperparameters = json.load(f)

    logger.info("Hyperparameters: {}".format(hyperparameters))

    metadata = read_metadata(hyperparameters['metadata_path'])
    logger.info("Loaded metadata")

    train_dataset = CriteoParquetDataset(hyperparameters['data_path']['train'])
    valid_dataset = CriteoParquetDataset(hyperparameters['data_path']['validation'])

    logger.info("Loaded datasets")

    model_parameters = ModelParameters(
        dense_input_feature_size=hyperparameters['dense_input_feature_size'],
        sparse_embedding_sizes=hyperparameters['sparse_embedding_sizes'],
        dense_output_size=hyperparameters['dense_output_size'],
        sparse_output_size=hyperparameters['sparse_output_size'],
        dense_hidden_size=hyperparameters['dense_hidden_size'],
        sparse_hidden_size=hyperparameters['sparse_hidden_size'],
        prediction_hidden_size=hyperparameters['prediction_hidden_size']
    )

    model = DLRM(metadata=metadata, parameters=model_parameters).to(hyperparameters['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])

    # Binary Cross Entropy loss
    criterion = nn.BCELoss()

    # DataLoader for your dataset
    train_loader = iter(DataLoader(train_dataset, batch_size=hyperparameters['batch_size']['train'], shuffle=True))
    valid_loader = iter(DataLoader(valid_dataset, batch_size=hyperparameters['batch_size']['validation'], shuffle=False))

    # Number of epochs
    num_epochs = hyperparameters['num_epochs']

    # Initialize the best validation loss to a high value
    best_valid_loss = float('inf')

    # Training Loop
    for epoch in range(num_epochs):
        logger.info("Epoch: {}".format(epoch + 1))

        start = time.time()
        # Training Phase
        train_loss = 0
        model.train()
        for batch_idx in tqdm(range(hyperparameters['batches_per_epoch']), ncols=80):
            labels, dense, sparse = next(train_loader)
            labels = labels.to(hyperparameters['device'])
            dense = dense.to(hyperparameters['device'])
            sparse = sparse.to(hyperparameters['device'])

            # Forward pass
            outputs = model(dense, sparse)
            loss = criterion(outputs, labels)

            # logger.info("--- Loss: {}".format(loss.item()))

            # Backward pass and optimization
            optimizer.zero_grad()
            # logger.info("Zeroed gradients")
            loss.backward()
            # logger.info("Backward pass done")
            optimizer.step()
            # logger.info("Optimizer step done")

            # logger.info("--- Backward pass and optimization done")
            train_loss = train_loss + (
                    (loss.item() - train_loss) / (batch_idx + 1))

        logger.info("Train Time taken: {:.2f}s".format(time.time() - start))

        start = time.time()
        # Validation Phase
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for batch_idx in tqdm(range(hyperparameters['batches_per_epoch']), ncols=80):
                labels, dense, sparse = next(valid_loader)
                # Move data to the appropriate device
                labels = labels.to(hyperparameters['device'])
                dense = dense.to(hyperparameters['device'])
                sparse = sparse.to(hyperparameters['device'])

                # Forward pass
                outputs = model(dense, sparse)
                loss = criterion(outputs, labels)
                valid_loss = valid_loss + (
                        (loss.item() - valid_loss) / (batch_idx + 1))

        logger.info("Validation Time taken: {:.2f}s".format(time.time() - start))
        logger.info("----------------------------------------------")
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Valid Loss: {valid_loss:.4f}')
        logger.info("----------------------------------------------")

        # If the current validation loss is less than the best validation loss,
        # save the model and update the best validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), hyperparameters['model_path'])
            logger.info(f'Validation loss decreased. Saving model...')


if __name__ == '__main__':
    main()
