from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from scipy.optimize import rosen, rosen_der

class Evaluation(NamedTuple):
    f: np.ndarray
    g: np.ndarray | None

class Algorithm(ABC):
    def __init__(self, x: np.ndarray):
        self.x = x

    @abstractmethod
    def step(self, objective: Callable[[np.ndarray, bool], Evaluation]):
        ...

class Rosenbrock:
    def __init__(self, n: int = 2):
        self.n = n
        if n == 2:
            self.x = np.array((-1.1, 2.5))
        else:
            self.x = np.repeat(np.array([-1.2, 1.]), repeats=n//2)

        self.n_passes = 0 # number of forward and backward passes
        self.max_passes = 10*n

        self.f_best = float('inf')
        self.x_best = None

    def _update_f_best(self, x, f):
        if f < self.f_best:
            self.x_best = x
            self.f_best = f

    def __call__(self, x: np.ndarray, backward:bool):
        f = rosen(x)

        # update best x if within 10*n passes limit
        if self.n_passes < self.max_passes:
            self._update_f_best(x, f)

        self.n_passes += 1 # forward pass

        g = None
        if backward:
            g = rosen_der(x)
            self.n_passes += 1 # backward pass

        if self.n_passes < self.max_passes:
            print(f"{self.n_passes}: Evaluated {x}. {f = }, {g = }")

        return Evaluation(f, g)

    def minimize(self, algorithm: Algorithm):
        for _ in range(self.max_passes):
            algorithm.step(self)

            # algorithm can perform more than 1 pass per step, 1 is lower bound
            if self.n_passes >= self.max_passes:
                break

        print(f"reached {self.f_best} at {self.x_best}")
        if self.f_best > 1e-6:
            print(f"{algorithm.__class__.__name__} failed to minimize rosenbrock in {self.max_passes} passes")

        else:
            print(f'{algorithm.__class__.__name__} successfully minimized rosenbrock in under {self.max_passes} passes")
