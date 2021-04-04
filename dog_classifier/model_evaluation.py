import pandas as pd
import numpy as np
import altair as alt

from dog_classifier.breed_classifier import NeuralNet


# %%
def model_losses():
    train_losses = [4.828373968033565, 4.561895974477133, 4.3049539429800845,
                    4.10613343602135, 3.9616453170776356, 3.837490134012132,
                    3.729485934121267, 3.6096336637224464, 3.4845925603594092,
                    3.390888084684101, 3.2799783706665036, 3.20562988917033,
                    3.072563396181379, 2.9623924732208247, 2.870406493686495,
                    2.7523970808301663, 2.665678980236962, 2.535139397212437,
                    2.430639664332072, 2.333072783833458]
    validation_losses = [4.704964978354318, 4.440237283706664,
                         4.318263803209577,
                         4.110924073628017, 4.011837703841073,
                         3.990762727601187,
                         3.8492509978158136, 3.8797887223107472,
                         3.911121691976275,
                         3.717871563775199, 3.5882019826344083,
                         3.6028132949556624,
                         3.6062802246638705, 3.741273845945086, 3.6166011095047,
                         3.5896864277975893, 3.968828797340393,
                         3.668894120625087,
                         3.558329514094762, 3.6221354859215875]
    return np.array(train_losses), np.array(validation_losses)


# %%
def plot_losses(train_loss, valid_loss):
    train_loss_df = pd.DataFrame(dict(epochs=np.arange(len(train_loss)),
                                      loss=train_loss,
                                      key="Training"))
    valid_loss_df = pd.DataFrame(dict(epochs=np.arange(len(valid_loss)),
                                      loss=valid_loss, key="Validation"))
    data = pd.concat([train_loss_df, valid_loss_df])
    chart = alt.Chart(data).mark_line(point=True).encode(
        x='epochs',
        y='loss',
        color="key"
    ).properties(
        width=800,
        height=400,
        title="Loss Function vs Epoch"
    )
    chart.show()


# %%

train_loss, valid_loss = model_losses()
plot_losses(train_loss, valid_loss)

# %%
