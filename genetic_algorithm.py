from enum import Enum
import numpy as np
from multiprocessing import Pool
from functools import partial
from game import *

class SelectionMethod(Enum):
    ROULETTE = "roulette"
    RANKING = "ranking"

class CrossoverMethod(Enum):
    SINGLE_POINT = "single_point"
    UNIFORM = "uniform"

class FitnessMethod(Enum):
    FINAL_STATE = "final_state"
    STABILITY = "stability"

def run_pattern_with_history(initial_pattern, iterations=100):
    game = GameOfLife(size=1000, initial_size=initial_pattern.shape[0])
    start_row = (game.size - initial_pattern.shape[0]) // 2
    start_col = (game.size - initial_pattern.shape[1]) // 2
    game.grid[start_row:start_row+initial_pattern.shape[0], 
              start_col:start_col+initial_pattern.shape[1]] = initial_pattern
    
    cell_counts = []
    cell_counts.append(np.sum(game.grid))
    
    for _ in range(iterations):
        game.compute_next_state()
        cell_counts.append(np.sum(game.grid))
    
    return game.grid, np.array(cell_counts)

def evaluate_final_state_fitness(pattern, iterations=100):
    final_state, _ = run_pattern_with_history(pattern, iterations=iterations)
    return np.sum(final_state)

def evaluate_stability_fitness(pattern, iterations=100):
    _, cell_counts = run_pattern_with_history(pattern, iterations=iterations)
    
    mean_count = np.mean(cell_counts)
    min_count = np.min(cell_counts)
    
    stability = 1 / (1 + np.std(cell_counts))
    persistence = min_count / mean_count if mean_count > 0 else 0
    
    fitness = mean_count * (0.4 + 0.3 * stability + 0.3 * persistence)
    
    return fitness

class GeneticLifeOptimizer:
    def __init__(self, population_size=20, pattern_size=10, num_processes=4, 
                 selection_method=SelectionMethod.RANKING,
                 crossover_method=CrossoverMethod.UNIFORM,
                 fitness_method=FitnessMethod.FINAL_STATE):
        self.population_size = population_size
        self.pattern_size = pattern_size
        self.num_processes = num_processes
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.fitness_method = fitness_method
        self.population = [np.random.randint(0, 2, size=(pattern_size, pattern_size),
                          dtype=np.int8) for _ in range(population_size)]
        self.fitness_history = []
    
    def evaluate_fitness(self, pattern, iterations=100):
        if self.fitness_method == FitnessMethod.FINAL_STATE:
            return evaluate_final_state_fitness(pattern, iterations)
        else:
            return evaluate_stability_fitness(pattern, iterations)
    
    def parallel_fitness(self, population, iterations=100):
        with Pool(processes=self.num_processes) as pool:
            fitness_scores = pool.map(
                partial(self.evaluate_fitness, iterations=iterations), 
                population
            )
        return np.array(fitness_scores)

    def select_parents_roulette(self, fitness_scores):
        total_fitness = np.sum(fitness_scores)
        if total_fitness <= 0:
            return np.random.choice(len(fitness_scores), size=2, replace=False)
            
        probabilities = fitness_scores / total_fitness
        first_parent = np.random.choice(len(fitness_scores), p=probabilities)
        
        remaining_probs = probabilities.copy()
        remaining_probs[first_parent] = 0
        remaining_probs = remaining_probs / np.sum(remaining_probs)
        second_parent = np.random.choice(len(fitness_scores), p=remaining_probs)
            
        return [first_parent, second_parent]

    def select_parents_ranking(self, fitness_scores):
        ranks = np.argsort(np.argsort(fitness_scores))
        ranks = ranks + 1  
        
        total_rank = np.sum(ranks)
        probabilities = ranks / total_rank
        
        first_parent = np.random.choice(len(fitness_scores), p=probabilities)
        
        remaining_probs = probabilities.copy()
        remaining_probs[first_parent] = 0
        remaining_probs = remaining_probs / np.sum(remaining_probs)
        second_parent = np.random.choice(len(fitness_scores), p=remaining_probs)
        
        return [first_parent, second_parent]
    
    def select_parents(self, fitness_scores):
        if self.selection_method == SelectionMethod.ROULETTE:
            return self.select_parents_roulette(fitness_scores)
        else:
            return self.select_parents_ranking(fitness_scores)
    
    def single_point_crossover(self, parent1, parent2):
        flat_size = self.pattern_size * self.pattern_size
        p1_flat = parent1.flatten()
        p2_flat = parent2.flatten()
        crossover_point = np.random.randint(0, flat_size)
        child1 = np.concatenate([p1_flat[:crossover_point], 
                               p2_flat[crossover_point:]]).reshape((self.pattern_size, self.pattern_size))
        child2 = np.concatenate([p2_flat[:crossover_point], 
                               p1_flat[crossover_point:]]).reshape((self.pattern_size, self.pattern_size))
        return child1, child2
    
    def uniform_crossover(self, parent1, parent2):
        mask = np.random.random(parent1.shape) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
    
    def crossover(self, parent1, parent2):
        if self.crossover_method == CrossoverMethod.SINGLE_POINT:
            return self.single_point_crossover(parent1, parent2)
        else:
            return self.uniform_crossover(parent1, parent2)
    
    def mutate(self, pattern, mutation_rate=0.1):
        mutation_mask = np.random.random(pattern.shape) < mutation_rate
        pattern[mutation_mask] = 1 - pattern[mutation_mask]
        return pattern
    
    def evolve(self, generations=20, iterations=100):
        best_pattern = None
        best_fitness = 0
        self.fitness_history = []
        
        for gen in range(generations):
            fitness_scores = self.parallel_fitness(self.population, iterations)
            self.fitness_history.append({
                'max': np.max(fitness_scores),
                'mean': np.mean(fitness_scores),
                'scores': fitness_scores.copy()
            })
            
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_pattern = self.population[gen_best_idx].copy()
            
            new_population = []
            elite_idx = np.argsort(fitness_scores)[-2:]  # Keep 2 best patterns
            new_population.extend([self.population[i].copy() for i in elite_idx])
            
            while len(new_population) < self.population_size:
                parent_indices = self.select_parents(fitness_scores)
                child1, child2 = self.crossover(
                    self.population[parent_indices[0]], 
                    self.population[parent_indices[1]]
                )
                new_population.extend([self.mutate(child1), self.mutate(child2)])
            
            self.population = new_population[:self.population_size]
        
        return best_pattern, best_fitness, self.fitness_history

def optimize_life_pattern(population_size=20, generations=20, iterations=100, 
                         num_processes=4, 
                         selection_method=SelectionMethod.RANKING,
                         crossover_method=CrossoverMethod.UNIFORM,
                         fitness_method=FitnessMethod.FINAL_STATE):
    optimizer = GeneticLifeOptimizer(
        population_size=population_size,
        pattern_size=10,
        num_processes=num_processes,
        selection_method=selection_method,
        crossover_method=crossover_method,
        fitness_method=fitness_method
    )
    return optimizer.evolve(generations=generations, iterations=iterations)