import numpy as np
from scipy.linalg import solve_banded
from ..base import Algorithm

class TridiagonalCubicSurrogate(Algorithm):
    def __init__(self, x: np.ndarray, n_probes: int = 2, epsilon: float = 1e-4):
        super().__init__(x)
        self.n = len(x)
        self.n_probes = n_probes
        self.epsilon = epsilon

    def _apply_B(self, B_diag, B_off, p):
        """Helper to compute B*p for a tridiagonal B."""
        Bp = np.zeros(self.n)
        Bp[0] = B_diag[0] * p[0] + B_off[0] * p[1]
        for i in range(1, self.n - 1):
            Bp[i] = B_off[i-1] * p[i-1] + B_diag[i] * p[i] + B_off[i] * p[i+1]
        Bp[self.n-1] = B_off[self.n-2] * p[self.n-2] + B_diag[self.n-1] * p[self.n-1]
        return Bp

    def step(self, objective):
        # 1. Evaluate current point
        f0, g0 = objective(self.x, backward=True).f, objective(self.x, backward=True).g

        if np.linalg.norm(g0) < 1e-6:
            return

        # 2. Build linear system to solve for model parameters
        num_unknowns = 2 * self.n
        A = []
        b = []
        
        for _ in range(self.n_probes):
            d = np.random.randn(self.n)
            d /= np.linalg.norm(d) if np.linalg.norm(d) > 0 else 1.0
            
            p_probe = self.epsilon * d
            f_probe, g_probe = objective(self.x + p_probe, backward=True).f, objective(self.x + p_probe, backward=True).g
            
            # Function value equation
            row_f = np.zeros(num_unknowns)
            row_f[:self.n] = 0.5 * p_probe**2
            row_f[self.n:-1] = p_probe[:-1] * p_probe[1:]
            row_f[-1] = (1/6) * np.sum(p_probe**3)
            A.append(row_f)
            b.append(f_probe - f0 - np.dot(g0, p_probe))

            # Gradient value equations
            for i in range(self.n):
                row_g = np.zeros(num_unknowns)
                row_g[i] = p_probe[i]
                if i > 0: row_g[self.n + i - 1] = p_probe[i-1]
                if i < self.n - 1: row_g[self.n + i] = p_probe[i+1]
                row_g[-1] = 0.5 * p_probe[i]**2
                A.append(row_g)
                b.append(g_probe[i] - g0[i])
        
        # 3. Solve for model and find step
        p = -g0 / (np.linalg.norm(g0) if np.linalg.norm(g0) > 0 else 1) # Default step
        try:
            h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            B_diag, B_off, c = h[:self.n], h[self.n:-1], h[-1]
            
            B_banded = np.zeros((3, self.n))
            B_banded[0, 1:] = B_off
            B_banded[1, :] = B_diag
            B_banded[2, :-1] = B_off
            B_banded[1, :] += 1e-4 # Regularization

            p_dir = solve_banded((1, 1), B_banded, -g0, check_finite=False)
            
            # Minimize 1D cubic surrogate along p_dir
            c3 = (c / 6) * np.sum(p_dir**3)
            c2 = 0.5 * np.dot(p_dir, self._apply_B(B_diag, B_off, p_dir))
            c1 = np.dot(g0, p_dir)
            
            # Stationary points of m'(a) = 3*c3*a^2 + 2*c2*a + c1 = 0
            alpha = 1.0
            if abs(3*c3) > 1e-8:
                discriminant = (2*c2)**2 - 4*(3*c3)*c1
                if discriminant >= 0:
                    sqrt_disc = np.sqrt(discriminant)
                    a1 = (-2*c2 + sqrt_disc) / (6*c3)
                    a2 = (-2*c2 - sqrt_disc) / (6*c3)
                    
                    # Check second derivative for minimum (m''(a) > 0)
                    val1 = 2*c2 + 6*c3*a1 if np.isfinite(a1) else -1
                    val2 = 2*c2 + 6*c3*a2 if np.isfinite(a2) else -1

                    if val1 > 0 and 0 < a1 <= 2.0: alpha = a1
                    elif val2 > 0 and 0 < a2 <= 2.0: alpha = a2
            elif abs(c2) > 1e-8:
                alpha_lin = -c1 / (2*c2)
                if 0 < alpha_lin <= 2.0: alpha = alpha_lin
            
            p = alpha * p_dir

        except (np.linalg.LinAlgError, ValueError):
            pass

        # 4. Final step with safeguard
        if np.dot(g0, p) > 0: # Ensure descent direction
            p = -g0

        norm_p = np.linalg.norm(p)
        if norm_p > 10.0: # Prevent overly large steps
            p = 10.0 * p / norm_p
        
        self.x += p
