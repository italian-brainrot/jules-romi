from .base import Rosenbrock
from .algorithms.gd import GD
from .algorithms.stochastic_hill_climbing import StochasticHillClimbing

def main():
    """Tests if algorithm is able to minimize rosenbrock in 20 passes"""
    rosenbrock = Rosenbrock()
    algorithm = StochasticHillClimbing(rosenbrock.x)
    rosenbrock.minimize(algorithm)

if __name__ == "__main__":
    main()