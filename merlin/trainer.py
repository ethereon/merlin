from typing import Callable, List, Optional, Union

import tensorflow as tf
from tensorflow.python.training.tracking.base import Trackable
from tensorflow.python.training.tracking.util import add_variable

from merlin.optimizer import Optimizer
from merlin.runloop import TaskCompleted
from merlin.spec import Default, Spec
from merlin.util.collections import as_enumerable
from merlin.util.logging import get_logger

logger = get_logger(__name__)

VariableSource = Union[tf.Module, List[tf.Module], List[tf.Variable]]

ObjectiveFunction = Callable[[], tf.Tensor]


class Trainer(Trackable):

    class Config(Spec):
        # An optional name to identify the trainer.
        name: Optional[str] = None
        # Maximum number of training steps to execute.
        max_steps: Optional[int] = None
        # Clip the gradient norms.
        max_gradient_norm: Optional[float] = None
        # Log progress to console after this many steps.
        # Set to None to disable.
        log_progress_interval: int = 100
        # Optimizer parameters
        optimizer: Optimizer.Config = Default(Optimizer.Config())
        # Number of steps to take each time the trainer is called
        num_steps_per_call: int = 1

    def __init__(self,
                 variables: VariableSource,
                 objective: ObjectiveFunction,
                 config: Config = None):
        self.objective = objective
        self.config = config or Trainer.Config()
        self.optimizer = Optimizer(config=self.config.optimizer)
        self.get_variables = self._make_variables_extractor(variables)

        tf.Variable(0, name='num_completed_steps', dtype=tf.int64)
        self.loss = None

        # Setup trackable internal state that'll be saved/restored
        self._track_trackable(
            trackable=self.optimizer,
            name='optimizer'
        )
        with tf.device('/CPU:0'):
            self._num_completed_steps = add_variable(
                trackable=self,
                name='num_completed_steps',
                dtype=tf.int64,
                initializer=0,
                trainable=False
            )

    def take_single_step(self):
        """
        Performs a single training step.
        """
        if self.is_training_complete:
            raise TaskCompleted()

        # Forward pass
        with tf.GradientTape() as tape:
            self.loss = self.objective()

        # Backward pass
        self._backprop(tape=tape)

        # Update internal state
        self._maybe_log_progress()
        self._num_completed_steps.assign_add(1)

    def summarize_loss(self, name=None):
        assert self.loss is not None
        if name is None:
            name = 'training_loss'
            if self.config.name:
                name += '/' + self.config.name
        tf.summary.scalar(
            name=name,
            step=self.last_step_index,
            data=self.loss
        )

    @property
    def num_completed_steps(self):
        return int(self._num_completed_steps)

    @property
    def last_step_index(self):
        return self.num_completed_steps - 1

    @property
    def is_training_complete(self):
        return (
            (self.config.max_steps is not None) and
            (self.num_completed_steps >= self.config.max_steps)
        )

    def __call__(self, num_steps=None):
        for _ in range(num_steps or self.config.num_steps_per_call):
            self.take_single_step()

    def _backprop(self, tape):
        # Compute the gradients
        trainable_variables = self.get_variables()
        gradients = tape.gradient(target=self.loss, sources=trainable_variables)

        # Clip the gradients.
        if self.config.max_gradient_norm is not None:
            gradients, _ = tf.clip_by_global_norm(
                t_list=gradients,
                clip_norm=self.config_max_gradient_norm
            )

        # Optimize
        self.optimizer.apply_gradients(grads_and_vars=zip(gradients, trainable_variables))

    def _maybe_log_progress(self):
        if not (self.config.log_progress_interval and
                (self.num_completed_steps % self.config.log_progress_interval == 0)):
            return
        # Step count and loss
        msg = f'Steps = {self.num_completed_steps:<12} Loss = {self.loss:<12g}'
        # The trainer name if provided
        if self.config.name:
            msg += f'{self.config.name:>30}'
        logger.info(msg)

    def _make_variables_extractor(self, variable_source: VariableSource):
        """
        Returns a function that returns a list of trainable variables extracted
        from the given variable source(s).

        The returned function should be invoked after at least one invocation of
        all "parent" modules/layers to allow for deferred variables to be created.
        The trainer assumes that calling the objective function satisfies this requirement.
        """
        variables = []

        def get_variables() -> List[tf.Variable]:
            if not variables:
                for src in as_enumerable(variable_source):
                    if isinstance(src, tf.Module):
                        variables.extend(src.trainable_variables)
                    elif isinstance(src, tf.Variable):
                        variables.append(src)
                    else:
                        raise ValueError(f'Invalid variable source type: {type(src)}')
            if not variables:
                raise ValueError('No trainable variables found.')

            return variables

        return get_variables
