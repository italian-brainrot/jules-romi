import math

import numpy as np

from ..base import Algorithm


class StochasticHillClimbing(Algorithm):
    def __init__(self, x: np.ndarray, step_size=0.1):
        super().__init__(x)
        self.step_size = step_size
        self.f_best = None

    def step(self, objective):
        if self.f_best is None:
            self.f_best, _ = objective(self.x, False)

        candidate = self.x + np.random.normal(scale=self.step_size)
        f_c, _ = objective(candidate, False)

        if f_c < self.f_best:
            self.x = candidate
            self.f_best = f_c
