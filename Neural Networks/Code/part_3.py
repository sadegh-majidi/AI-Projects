import os

import matplotlib.pyplot as plt
import numpy as np
from keras.losses import mean_squared_error
from sklearn.model_selection import KFold
from tensorflow import keras

plt.style.use("ggplot")
os.environ['TF_KERAS'] = '1'


class DatasetCreator:

    def __init__(self, function, num_of_features, train_n_samples, train_min, train_max, test_n_samples, test_min, test_max):
        self.function = function
        self.num_of_features = num_of_features
        self.train_n_samples = train_n_samples
        self.train_min = train_min
        self.train_max = train_max
        self.test_n_samples = test_n_samples
        self.test_min = test_min
        self.test_max = test_max

    def get_train_data(self):
        X = np.random.uniform(self.train_min, self.train_max, (self.train_n_samples, self.num_of_features))
        labels = self.function(*X.T)
        p = np.random.permutation(self.train_n_samples)
        return X[p], labels[p]

    def get_test_data(self):
        X = np.random.uniform(self.test_min, self.test_max, (self.test_n_samples, self.num_of_features))
        labels = self.function(*X.T)
        return X, labels


class FunctionApprox:
    def __init__(self, dataset_creator: DatasetCreator, num_of_features: int, batch_size: int, num_epochs: int, k_folds: int = 5):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k_folds = k_folds
        self.dataset_creator = dataset_creator
        self.num_of_features = num_of_features
        self.loss_function = mean_squared_error
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.X, self.y = self.dataset_creator.get_train_data()
        self.model = None
        self.best_loss = None
        self.avg_loss = None

    def get_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(128, input_dim=self.num_of_features, activation='relu'))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
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
        ax = fig.gca(projection='3d')
        test_X, test_y = self.dataset_creator.get_test_data()
        pred_y = self.model.predict(test_X)
        test_y = test_y.reshape(len(test_X[:, 0]))
        pred_y = pred_y.reshape(len(test_X[:, 0]))
        ax.plot_trisurf(test_X[:, 0], test_X[:, 1], pred_y, alpha=0.6)
        ax.plot_trisurf(test_X[:, 0], test_X[:, 1], test_y, alpha=0.5)
        return fig


def test_func_1(x, y):
    return np.sin(x) + np.cos(y)


def test_func_2(x, y):
    return x * y


def test_func_3(x, y, z):
    return np.sin(x) * (y ** 2) + np.log(np.abs(z)) * 3 * (np.cos(y) ** 2)


if __name__ == '__main__':
    ds_creator = DatasetCreator(test_func_3, num_of_features=3,
                                train_n_samples=24000, train_min=(-5, -5, -5), train_max=(5, 5, 5),
                                test_n_samples=200000, test_min=(-5, -5, -5), test_max=(5, 5, 5))
    f_predictor = FunctionApprox(ds_creator, num_of_features=3, batch_size=32, num_epochs=6)
    f_predictor.train()
    fig = f_predictor.plot_result()
    plt.show()
    fig.savefig('part_3.png')
