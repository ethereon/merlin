from typing import Callable

from merlin.util import logging

logger = logging.get_logger(__name__)

Task = Callable[..., None]


class TaskCompleted(Exception):
    pass


class RunLoop:
    # TODO(saumitro): Track timing stats for each task!

    def __init__(self):
        self.tasks = []

    def __iadd__(self, task: Task):
        self.add_task(task)
        return self

    def add_task(self, task: Task):
        self.tasks.append(task)

    def remove_task(self, task):
        assert task in self.tasks
        self.tasks.remove(task)

    def _execute_run_loop(self):
        while self.tasks:
            for task in self.tasks:
                try:
                    task()
                except TaskCompleted:
                    self.remove_task(task)

    def run(self):
        try:
            self._execute_run_loop()
        except KeyboardInterrupt:
            logger.info('User interrupted. Shutting down.')


def run(*tasks):
    loop = RunLoop()
    for task in tasks:
        loop.add_task(task)
    loop.run()
