import numpy as np

from ..base import Algorithm

class BarzilaiBorwein(Algorithm):
    def __init__(self, x: np.ndarray):
        super().__init__(x)
        self.g_old = None
        self.x_old = None

    def step(self, objective):
        _, g = objective(self.x, True)
        
        if self.g_old is not None:
            s = self.x - self.x_old
            y = g - self.g_old
            
            # Barzilai-Borwein step size
            alpha = np.abs(s.dot(y) / y.dot(y))
            
            self.x -= alpha * g
        else:
            # Fallback to gradient descent for the first step
            self.x -= 0.001 * g
            
        self.x_old = np.copy(self.x)
        self.g_old = np.copy(g)