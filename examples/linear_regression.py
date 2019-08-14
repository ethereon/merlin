import tensorflow as tf

from merlin.runloop import run
from merlin.trainer import Trainer
from merlin.modules.affine import Affine


def synthetic_data_source(batch_size, input_dim, output_dim):
    """
    Infinite synthetic data source that yields (x, W*x + b + Gaussian noise) pairs,
    where the weigths and biases are randomly initialized.
    """
    weights = tf.random.uniform(shape=(input_dim, output_dim))
    biases = tf.random.uniform(shape=(output_dim,))
    while True:
        x = tf.random.uniform(shape=(batch_size, input_dim))
        y = x @ weights + biases
        yield x, y + 0.01 * tf.random.normal(shape=y.shape)


def train(batch_size=32, input_dim=3, output_dim=2):
    """
    Train a model to fit linear data by minimizing the mean squared error (mse).
    """
    # A synthetic data source
    data_source = synthetic_data_source(
        batch_size=batch_size,
        input_dim=input_dim,
        output_dim=output_dim
    )

    # The output is assumed to be an affine function of the input
    model = Affine(units=output_dim)

    # Minimize the mean squared error between the ground truth and the predictions
    def mse_objective():
        x, y_true = next(data_source)
        y_pred = model(x)
        return tf.math.reduce_mean(tf.math.square(y_true - y_pred))

    # Run the trainer
    run(Trainer(objective=mse_objective, variables=model))


if __name__ == "__main__":
    train()
