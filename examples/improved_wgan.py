"""
An implementation of a Wasserstein GAN with gradient penalty, as described in:

    Improved Training of Wasserstein GANs
    https://arxiv.org/pdf/1704.00028.pdf

Trains the GAN on a toy dataset of a mixture of Gaussians.
"""

from typing import Tuple
from collections import namedtuple

import tensorflow as tf

from merlin.gradient import compute_gradient
from merlin.modules.activation import Activation
from merlin.modules.affine import Affine
from merlin.modules.util import chain
from merlin.ops.norm import safe_norm
from merlin.spec import Spec, Default
from merlin.runloop import run
from merlin.snapshot import Snapshot
from merlin.summarizer import Summarizer, summarizer
from merlin.trainer import Optimizer, Trainer


class Normal(namedtuple('NormalBase', 'mean stddev')):
    """
    A normal distribution parameterized by its mean and standard deviation.
    """


class GanConfig(Spec):
    """
    Configuration and hyper-parameters for the WGAN model.
    """

    # Scaling factor for the gradient penalty. \lambda in the paper.
    penalty_scale: float = 0.1

    # Real distribution parameters
    real_data_params: Tuple[Normal] = (
        Normal(mean=1., stddev=0.5),
        Normal(mean=5., stddev=0.5),
        Normal(mean=10., stddev=0.5)
    )

    # The data batch size
    batch_size: int = 256

    # Number of discriminator steps before each generator step
    num_discriminator_steps: int = 5

    # Summarizer for visualizing distribution histograms
    summarizer: Summarizer.Config = Default(Summarizer.Config(
        # Run tensorboard on this directory to see the visualizations
        output_dir='/tmp/wgan-gp/summary',
    ))

    # Snapshot the training state periodically
    snapshot: Snapshot.Config = Default(Snapshot.Config(
        directory='/tmp/wgan-gp/snapshot'
    ))

    # Common trainer parameters
    trainer: Trainer.Config = Default(Trainer.Config(
        optimizer=Optimizer.Config(
            name='Adam',
            params=dict(
                lr=1e-4,
                beta_1=0.5,
                beta_2=0.9,
            )
        )
    ))


@chain
def simple_net(hidden_units=512, output_units=1):
    return (
        Affine(units=hidden_units),
        Activation('relu'),

        Affine(units=hidden_units),
        Activation('relu'),

        Affine(units=hidden_units),
        Activation('relu'),

        Affine(units=output_units)
    )


def gaussian_mixture_sampler(batch_size: int, params: Tuple[Normal]):
    group_sizes = [batch_size // len(params)] * (len(params) - 1)
    group_sizes.append(batch_size - sum(group_sizes))
    while True:
        group_sizes.append(group_sizes.pop(0))
        yield tf.concat(
            [
                tf.random.normal(shape=(size, 1), mean=mean, stddev=stddev)
                for size, (mean, stddev) in zip(group_sizes, params)
            ],
            axis=0
        )


def train(config=GanConfig()):
    generator = simple_net(name='generator')

    discriminator = simple_net(name='discriminator')

    # Sample from the normal distribution for the generator's input
    generator_data = gaussian_mixture_sampler(
        batch_size=config.batch_size,
        params=[(0., 1.)]
    )
    # Let the real data be a mixture of Gaussians
    real_data = gaussian_mixture_sampler(
        batch_size=config.batch_size,
        params=config.real_data_params
    )

    def compute_discriminator_loss():
        # Get a real data point
        real_sample = next(real_data)
        # Use the generator to sample a "fake" data point
        generator_output = generator(next(generator_data))
        # Use the discriminator to compute the likelihood of fake/real
        score_fake = discriminator(generator_output)
        score_real = discriminator(real_sample)

        # Compute the gradient penalty by first samping uniformly along straight lines
        # between the generated and real samples (equation 3 in the paper).
        epsilon = tf.random.uniform(shape=real_sample.shape)
        interp = epsilon * real_sample + (1 - epsilon) * generator_output

        # Compute the gradient wrt to this interpolated input
        grad = compute_gradient(lambda: discriminator(interp), interp)
        slopes = safe_norm(grad, axis=1, keepdims=True)
        gradient_penalty = (slopes - 1.0) ** 2

        return tf.reduce_mean(score_fake - score_real + config.penalty_scale * gradient_penalty)

    def compute_generator_loss():
        score_fake = discriminator(generator(next(generator_data)))
        return tf.reduce_mean(-score_fake)

    generator_trainer = Trainer(
        variables=generator,
        objective=compute_generator_loss,
        config=config.trainer.replace(name='Generator')
    )

    discriminator_trainer = Trainer(
        variables=discriminator,
        objective=compute_discriminator_loss,
        config=config.trainer.replace(
            name='Discriminator',
            num_steps_per_call=config.num_discriminator_steps
        )
    )

    @summarizer(config.summarizer)
    def visualize():
        # Histograms
        tf.summary.histogram(
            name='real_distribution',
            data=next(real_data),
            step=generator_trainer.last_step_index
        )
        tf.summary.histogram(
            name='generator_distribution',
            data=generator(next(generator_data)),
            step=generator_trainer.last_step_index
        )

        # Training losses
        discriminator_trainer.summarize_loss()
        generator_trainer.summarize_loss()

    # Create a snapshot for restoring / periodically saving the training state
    snapshot = Snapshot(
        config=config.snapshot,
        generator=generator,
        discriminator=discriminator,
        generator_trainer=generator_trainer,
        discriminator_trainer=discriminator_trainer
    )
    snapshot.maybe_restore_latest()

    # Start training
    run(
        discriminator_trainer,
        generator_trainer,
        visualize,
        snapshot
    )


if __name__ == "__main__":
    train()
