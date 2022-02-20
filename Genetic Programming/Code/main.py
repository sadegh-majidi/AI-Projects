import matplotlib.pyplot as plt
import numpy as np

from genetic import GPFunctionApproximation
from structs import Population

MAX_SIZE = 1
MAX_DEPTH = 20
BASE_FUNCTIONS = {1: ['sin', 'cos'], 2: ['+', '-', '*', '/', '^']}
TERMINAL_SET = ['x'] if MAX_SIZE == 1 else ['x'+str(i) for i in range(MAX_SIZE)]
DEPTH = 4


def f(x):
    return np.sin(x) * np.cos(x)


X = [[x] for x in np.arange(0, 10, 0.01)]
y = [[f(x[0])] for x in X]

pop = Population(1000, 20, BASE_FUNCTIONS, TERMINAL_SET, 6, MAX_DEPTH)
gp_approx = GPFunctionApproximation(pop, 1000, 100)
gp_approx.fit(X, y)
best = gp_approx.best
print(best.gen)
y_pred = [[best.eval(x)[0]] for x in X]
print(best)

plt.plot(X, y, color='b', dashes=[6, 2])
plt.plot(X, y_pred, color='r', dashes=[6, 3])
plt.show()
