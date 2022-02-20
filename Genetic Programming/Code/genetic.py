import random
import warnings

import numpy as np

from structs import Chromosome

warnings.filterwarnings("error")


def traversal(poz, chromosome):
    if chromosome.gen[poz] in chromosome.terminal_set:
        return poz + 1
    elif chromosome.gen[poz] in chromosome.func_set[1]:
        return traversal(poz + 1, chromosome)
    else:
        new_poz = traversal(poz + 1, chromosome)
        return traversal(new_poz, chromosome)


def mutate(chromosome):
    poz = np.random.randint(len(chromosome.gen))
    if chromosome.gen[poz] in chromosome.func_set[1] + chromosome.func_set[2]:
        if chromosome.gen[poz] in chromosome.func_set[1]:
            chromosome.gen[poz] = random.choice(chromosome.func_set[1])
        else:
            chromosome.gen[poz] = random.choice(chromosome.func_set[2])
    else:
        chromosome.gen[poz] = random.choice(chromosome.terminal_set)
    return chromosome


def tournament_selection(population, num_sel):
    sample = random.sample(population.list, num_sel)
    best = sample[0]
    for i in range(1, len(sample)):
        if population.list[i].fitness < best.fitness:
            best = population.list[i]

    return best


def crossover(mother, father, max_depth):
    child = Chromosome(mother.terminal_set, mother.func_set, mother.depth, None)
    start_m = np.random.randint(len(mother.gen))
    start_f = np.random.randint(len(father.gen))
    end_m = traversal(start_m, mother)
    end_f = traversal(start_f, father)
    child.gen = mother.gen[:start_m] + father.gen[start_f: end_f] + mother.gen[end_m:]
    if child.get_depth() > max_depth and random.random() > 0.2:
        child = Chromosome(mother.terminal_set, mother.func_set, mother.depth)
    return child


def get_best(population):
    best = population.list[0]
    for i in range(1, len(population.list)):
        if population.list[i].fitness < best.fitness:
            best = population.list[i]

    return best


def get_worst(population):
    worst = population.list[0]
    for i in range(1, len(population.list)):
        if population.list[i].fitness > worst.fitness:
            worst = population.list[i]

    return worst


def replace_worst(population, chromosome):
    worst = get_worst(population)
    if chromosome.fitness < worst.fitness:
        for i in range(len(population.list)):
            if population.list[i].fitness == worst.fitness:
                population.list[i] = chromosome
                break
    return population


class GPFunctionApproximation:
    def __init__(self, population, iterations, epoch_feedback=500):
        self.iterations = iterations
        self.epoch_feedback = epoch_feedback
        self.population = population
        self.best = None

    def fit(self, X, y):
        for i in range(len(self.population.list)):
            self.population.list[i].calculate_fitness(X, y)
        for i in range(self.iterations):
            if i % self.epoch_feedback == 0:
                best_so_far = get_best(self.population)
                print("Best function: {0}".format(best_so_far.gen))
                print("Best fitness: {0}".format(best_so_far.fitness))

            mother = tournament_selection(self.population, self.population.num_selected)
            father = tournament_selection(self.population, self.population.num_selected)
            child = crossover(mother, father, self.population.max_depth)
            child = mutate(child)
            child.calculate_fitness(X, y)
            self.population = replace_worst(self.population, child)

        self.best = get_best(self.population)
