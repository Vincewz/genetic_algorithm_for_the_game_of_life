# Genetic Algorithm for Conway's Game of Life

This project implements a Genetic Algorithm to optimize patterns in Conway's Game of Life. It includes:
- Simulation of the Game of Life grid.
- Optimization using genetic operators like selection, crossover, and mutation.
- Fitness evaluation based on pattern stability or final state.

## Features
- Supports multiple selection and crossover methods.
- Configurable population size, pattern size, and number of generations.
- Interactive Game of Life visualization.

## Requirements
- Python 3.8+
- NumPy
- Matplotlib
- Multiprocessing

Install dependencies with:
```bash
pip install numpy matplotlib
```

## How to Use

### Run the Genetic Algorithm
Modify the `optimize_life_pattern` function in `genetic_algorithm.py` to configure parameters like:
- Population size
- Number of generations
- Selection, crossover, and fitness methods

### Example Usage
Below is a code snippet to test the implementation:

```python
from genetic_algorithm import optimize_life_pattern, SelectionMethod, CrossoverMethod, FitnessMethod

# Configure and run the optimizer
best_pattern, best_fitness, fitness_history = optimize_life_pattern(
    population_size=50,
    generations=100,
    iterations=200,
    num_processes=4,
    selection_method=SelectionMethod.RANKING,
    crossover_method=CrossoverMethod.UNIFORM,
    fitness_method=FitnessMethod.STABILITY
)

# Display the best pattern and fitness
print("Best Fitness:", best_fitness)
print("Best Pattern:\n", best_pattern)

# Visualize the optimized pattern in the Game of Life
from game import GameOfLife
game = GameOfLife(size=100)
game.set_pattern(best_pattern)
game.simulate()
```

### Run the Game of Life Simulator
You can also run the Game of Life independently:

```python
from game import GameOfLife
import numpy as np

# Create a random initial pattern
pattern = np.random.randint(0, 2, size=(20, 20))

# Initialize the game
game = GameOfLife(size=100)
game.set_pattern(pattern)

# Launch interactive simulation
game.simulate()
```
```
