import numpy as np

from ..base import Algorithm

class SeparableCubicSurrogate(Algorithm):
    def __init__(self, x: np.ndarray):
        super().__init__(x)
        self.eval_points = []
        self.eval_values = []
        self.eval_grads = []
        self.step_count = 0
        n = len(x)
        
        # Determine the number of points needed
        self.n_eval_points = 10
        
        # Generate evaluation points
        self.eval_points.append(np.copy(x))
        for i in range(self.n_eval_points - 1):
            point = np.copy(x)
            point[i % n] += 0.1  # Perturb different dimensions
            self.eval_points.append(point)

    def step(self, objective):
        if self.step_count < self.n_eval_points:
            # Data collection phase
            x_eval = self.eval_points[self.step_count]
            f, g = objective(x_eval, True)
            self.eval_values.append(f)
            self.eval_grads.append(g)
            self.step_count += 1
            if self.step_count == self.n_eval_points:
                self._fit_and_minimize()
        else:
            # After fitting, do nothing
            pass

    def _fit_and_minimize(self):
        n = len(self.x)
        n_points = len(self.eval_points)
        
        # Total number of equations: n_points * (n + 1)
        # Total number of variables: 4 * n + 1
        A = np.zeros((n_points * (n + 1), 4 * n + 1))
        b = np.zeros(n_points * (n + 1))

        for i in range(n_points):
            x = self.eval_points[i]
            f = self.eval_values[i]
            g = self.eval_grads[i]
            
            # Equation for function value
            row = np.zeros(4 * n + 1)
            row[0] = 1
            for j in range(n):
                row[1 + 4*j : 1 + 4*(j+1)] = [x[j]**4, x[j]**3, x[j]**2, x[j]]
            A[i*(n+1)] = row
            b[i*(n+1)] = f
            
            # Equations for gradient values
            for k in range(n):
                row = np.zeros(4 * n + 1)
                row[1 + 4*k : 1 + 4*(k+1)] = [4*x[k]**3, 3*x[k]**2, 2*x[k], 1]
                A[i*(n+1) + k + 1] = row
                b[i*(n+1) + k + 1] = g[k]
        
        # Solve for model coefficients
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        C = coeffs[0]
        model_coeffs = coeffs[1:].reshape((n, 4))
        
        # Minimize the surrogate model
        x_min = np.zeros(n)
        for i in range(n):
            # Derivative of the quartic polynomial
            deriv_coeffs = [4*model_coeffs[i, 0], 3*model_coeffs[i, 1], 2*model_coeffs[i, 2], model_coeffs[i, 3]]
            roots = np.roots(deriv_coeffs)
            
            # Find the real root that minimizes the quartic
            real_roots = roots[np.isreal(roots)].real
            if len(real_roots) > 0:
                f_vals = np.polyval(model_coeffs[i], real_roots)
                x_min[i] = real_roots[np.argmin(f_vals)]
            else:
                x_min[i] = self.x[i]

        self.x = x_min
