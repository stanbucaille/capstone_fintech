from datetime import datetime


class Clock(object):

    def __init__(self):
        self._start = 0
        self._step = self._start

    @property
    def start(self) -> int:
        return self._start

    @property
    def step(self) -> int:
        return self._step

    def now(self) -> str:
        return datetime.now()

    def now_formatted(self) -> str:
        return self.now().strftime("%H:%M:%S")

    def increment(self, step_size: int = 1):
        self._step += step_size

    def reset(self):
        self._step = self._start
