import random
import warnings

import numpy as np

warnings.filterwarnings("error")


class Chromosome:
    def __init__(self, terminal_set, func_set, depth, method='full'):
        self.depth = depth
        self.gen = []
        self.terminal_set = terminal_set
        self.func_set = func_set
        self.fitness = None
        self.make_chromosome(method=method)

    def make_chromosome(self, level: int = 0, method: str = 'full'):
        if level == self.depth:
            self.gen.append(random.choice(self.terminal_set))
        elif method == 'full':
            val = random.choice(self.func_set[1] + self.func_set[2])
            if val in self.func_set[2]:
                self.gen.append(random.choice(self.func_set[2]))
                self.make_chromosome(level + 1)
                self.make_chromosome(level + 1)
            else:
                self.gen.append(random.choice(self.func_set[1]))
                self.make_chromosome(level + 1)
        else:
            if random.random() > 0.3:
                val = random.choice(self.func_set[2] + self.func_set[1])
                if val in self.func_set[2]:
                    self.gen.append(val)
                    self.make_chromosome(level + 1, 'grow')
                    self.make_chromosome(level + 1, 'grow')
                else:
                    self.gen.append(val)
                    self.make_chromosome(level + 1, 'grow')
            else:
                val = random.choice(self.terminal_set)
                self.gen.append(val)

    def eval(self, inp, poz=0):
        if self.gen[poz] in self.terminal_set:
            return inp[int(self.gen[poz][1:])], poz
        elif self.gen[poz] in self.func_set[2]:
            poz_op = poz
            left, poz = self.eval(inp, poz + 1)
            right, poz = self.eval(inp, poz + 1)
            if self.gen[poz_op] == '+':
                return left + right, poz
            elif self.gen[poz_op] == '-':
                return left - right, poz
            elif self.gen[poz_op] == '*':
                return left * right, poz
            elif self.gen[poz_op] == '^':
                return left ** right, poz
            elif self.gen[poz_op] == '/':
                return left / right, poz
        else:
            poz_op = poz
            left, poz = self.eval(inp, poz + 1)
            if self.gen[poz_op] == 'sin':
                return np.sin(left), poz
            elif self.gen[poz_op] == 'cos':
                return np.cos(left), poz

    def calculate_fitness(self, inputs, outputs):
        diff = 0
        for i in range(len(inputs)):
            try:
                diff += (self.eval(inputs[i])[0] - outputs[i][0]) ** 2
            except:
                self.gen = []
                if random.random() > 0.5:
                    self.make_chromosome(method='grow')
                else:
                    self.make_chromosome()
                self.calculate_fitness(inputs, outputs)

        if len(inputs) == 0:
            return 1e9
        self.fitness = diff / (len(inputs))
        return self.fitness

    def __get_depth_aux(self, poz=0):
        elem = self.gen[poz]

        if elem in self.func_set[2]:
            left, poz = self.__get_depth_aux(poz + 1)
            right, poz = self.__get_depth_aux(poz)

            return 1 + max(left, right), poz
        elif elem in self.func_set[1]:
            left, poz = self.__get_depth_aux(poz + 1)
            return left + 1, poz
        else:
            return 1, poz + 1

    def get_depth(self):
        return self.__get_depth_aux()[0] - 1

    def print(self, gen: list):
        if gen[0] in self.func_set[1]:
            f, gen = gen[0], gen[1:]
            x, gen = self.print(gen)
            return f'{f}({x})', gen
        elif gen[0] in self.func_set[2]:
            f, gen = gen[0], gen[1:]
            x, gen = self.print(gen)
            x1, gen = self.print(gen)
            return f'({x} {f} {x1})', gen
        else:
            x, gen = gen[0], gen[1:]
            return f'{x}', gen

    def __str__(self):
        return self.print(self.gen.copy())[0]


class Population:
    def __init__(self, size, num_selected, func_set, terminal_set, depth, max_depth):
        self.size = size
        self.num_selected = num_selected
        self.list = Population.create_population(self.size, func_set, terminal_set, depth)
        self.max_depth = max_depth

    @staticmethod
    def create_population(number, func_set, terminal_set, depth):
        pop_list = []
        for i in range(number):
            if random.random() > 0.5:
                pop_list.append(Chromosome(terminal_set, func_set, depth, 'grow'))
            else:
                pop_list.append(Chromosome(terminal_set, func_set, depth, 'full'))
        return pop_list
