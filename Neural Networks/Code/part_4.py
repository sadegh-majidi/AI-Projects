import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.losses import mean_squared_error
from sklearn.model_selection import KFold
from tensorflow import keras

plt.style.use("ggplot")
os.environ['TF_KERAS'] = '1'


class FunctionApprox:
    def __init__(self, X, y, batch_size: int, num_epochs: int, k_folds: int = 5):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k_folds = k_folds
        self.loss_function = mean_squared_error
        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.X, self.y = X, y
        self.model = None
        self.best_loss = None
        self.avg_loss = None

    def get_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(50, input_dim=1, activation='relu'))
        model.add(keras.layers.Dense(50, activation='relu'))
        model.add(keras.layers.Dense(1))

        model.compile(loss=self.loss_function, optimizer=self.optimizer)
        return model

    def train(self):
        loss_history = []
        model_history = []
        k_fold = KFold(n_splits=self.k_folds, shuffle=True)

        for train, test in k_fold.split(self.X, self.y):
            model = self.get_model()
            model.fit(self.X[train], self.y[train], batch_size=self.batch_size, epochs=self.num_epochs, verbose=True)
            scores = model.evaluate(self.X[test], self.y[test], verbose=0)
            loss_history.append(scores)
            model_history.append(model)

        for i in range(0, len(loss_history)):
            print('****************************************************************************')
            print(f' fold {i + 1} - Loss: {loss_history[i]}')
        print('****************************************************************************')
        print(f' Average Loss: {np.mean(loss_history)}')
        print('****************************************************************************')

        best_model_index = np.argmin(loss_history)
        best_model = model_history[best_model_index]
        self.model = best_model
        self.best_loss = loss_history[best_model_index]
        self.avg_loss = np.mean(loss_history)
        return best_model

    def plot_result(self):
        fig = plt.figure()
        test_X = np.linspace(self.X.min() - 10, self.X.max() + 10, 400)
        pred_y = self.model.predict(test_X)
        plt.plot(test_X, pred_y, '-r')
        plt.plot(self.X, self.y, '--b', alpha=0.5)
        return fig


if __name__ == '__main__':
    df = pd.read_csv('function_pt4.csv')
    f_predictor = FunctionApprox(X=df['x'].to_numpy(), y=df['y'].to_numpy(), batch_size=32, num_epochs=450)
    f_predictor.train()
    fig = f_predictor.plot_result()
    plt.show()
    fig.savefig('part_4.png')
