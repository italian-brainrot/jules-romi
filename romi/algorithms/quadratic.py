import numpy as np

from ..base import Algorithm

class QuadraticSurrogate(Algorithm):
    def __init__(self, x: np.ndarray):
        super().__init__(x)
        self.eval_points = []
        self.eval_values = []
        self.eval_grads = []
        self.step_count = 0
        n = len(x)
        
        # Determine the number of points needed
        self.n_coeffs = 1 + n + n * (n + 1) // 2
        self.n_eval_points = int(np.ceil(self.n_coeffs / (n + 1)))
        
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
        
        A = np.zeros((n_points * (n + 1), self.n_coeffs))
        b = np.zeros(n_points * (n + 1))

        for i in range(n_points):
            x = self.eval_points[i]
            f = self.eval_values[i]
            g = self.eval_grads[i]
            
            # Equation for function value
            row = np.zeros(self.n_coeffs)
            row[0] = 1
            row[1:n+1] = x
            
            p = n + 1
            for j in range(n):
                for k in range(j, n):
                    if j == k:
                        row[p] = 0.5 * x[j]**2
                    else:
                        row[p] = x[j] * x[k]
                    p += 1
            A[i*(n+1)] = row
            b[i*(n+1)] = f
            
            # Equations for gradient values
            for l in range(n):
                row = np.zeros(self.n_coeffs)
                row[1+l] = 1
                p = n + 1
                for j in range(n):
                    for k in range(j, n):
                        if j == l and k == l:
                            row[p] = x[j]
                        elif j == l:
                            row[p] = x[k]
                        elif k == l:
                            row[p] = x[j]
                        p += 1
                A[i*(n+1) + l + 1] = row
                b[i*(n+1) + l + 1] = g[l]
        
        # Solve for model coefficients
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        c = coeffs[0]
        g0 = coeffs[1:n+1]
        
        H = np.zeros((n, n))
        p = n + 1
        for j in range(n):
            for k in range(j, n):
                H[j, k] = coeffs[p]
                H[k, j] = coeffs[p]
                p += 1
        
        # Minimize the surrogate model
        try:
            x_min = np.linalg.solve(H, -g0)
            self.x = x_min
        except np.linalg.LinAlgError:
            # If the Hessian is singular, we can't solve the system.
            # Fallback to the last evaluated point.
            self.x = self.eval_points[-1]
