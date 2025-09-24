import math

import numpy as np

from ..base import Algorithm


class GD(Algorithm):
    def __init__(self, x: np.ndarray, step_size=1, c=1e-4):
        super().__init__(x)
        self.step_size = step_size
        self.c = c

    def step(self, objective):
        f_0, g = objective(self.x, True)
        d = g.dot(g)

        a = self.step_size
        for i in range(20):
            f_a, _ = objective(self.x - g*a, False)

            if f_a < f_0 - self.c * a * max(d, 0):
                self.x = self.x - g*a
                return

            a = a * 0.5
