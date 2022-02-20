import os

import matplotlib.pyplot as plt
import numpy as np
import skimage
from keras.losses import mean_squared_error
from sklearn.model_selection import KFold
from tensorflow import keras

plt.style.use("ggplot")
os.environ['TF_KERAS'] = '1'


class NoisyDataset:
    LOW, MEDIUM, HIGH, VERY_HIGH = 0, 1, 2, 3

    def __init__(self, noise_level, train_n_samples, test_n_samples) -> None:
        self.noise_level = noise_level
        self.train_n_samples = train_n_samples
        self.test_n_samples = test_n_samples
        try:
            X_train, _, _, _ = keras.datasets.mnist.load_data()
        except:
            with np.load('mnist.npz', allow_pickle=True) as f:
                X_train = f['x_train']

        X_train = X_train.astype('float32')
        X_train /= 255

        p = np.random.permutation((len(X_train)))
        all_img = X_train[p]
        all_img = all_img[0: self.train_n_samples + self.test_n_samples]
        self.train_labels = all_img[0:self.train_n_samples]
        self.test_labels = all_img[self.train_n_samples: self.train_n_samples+self.test_n_samples]
        noisy_images = np.array(list(map(lambda x: self.make_noisy(x), all_img))).reshape(-1, 28, 28)

        self.train_images = noisy_images[0:self.train_n_samples]
        self.test_images = noisy_images[self.train_n_samples: self.train_n_samples + self.test_n_samples]

        self.train_images = self.train_images.reshape(self.train_images.shape[0], -1)
        self.test_images = self.test_images.reshape(self.test_images.shape[0], -1)
        self.train_labels = self.train_labels.reshape(self.train_labels.shape[0], -1)
        self.test_labels = self.test_labels.reshape(self.test_labels.shape[0], -1)

    def make_noisy(self, image):
        res = image
        if self.noise_level >= NoisyDataset.LOW:
            res = skimage.util.random_noise(image)
        if self.noise_level >= NoisyDataset.MEDIUM:
            res = skimage.util.random_noise(res)
            res = skimage.util.random_noise(res)
        if self.noise_level >= NoisyDataset.HIGH:
            res = skimage.util.random_noise(res)
            res = skimage.util.random_noise(res, clip=False)
        if self.noise_level >= NoisyDataset.VERY_HIGH:
            res = skimage.util.random_noise(res, mode='s&p', clip=False)

        return res

    def get_train_dataset(self):
        return self.train_images, self.train_labels

    def get_test_dataset(self):
        return self.test_images, self.test_labels


class DigitImageDeNoise:
    def __init__(self, batch_size: int, num_epochs: int, k_folds: int = 5):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k_folds = k_folds
        self.loss_function = mean_squared_error
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.model = None
        self.best_loss = None
        self.avg_loss = None

    def get_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, input_dim=784, activation='relu'))
        model.add(keras.layers.Dense(784, activation='sigmoid'))

        model.compile(loss=self.loss_function, optimizer=self.optimizer)
        return model

    def train(self, X, y):
        loss_history = []
        model_history = []
        k_fold = KFold(n_splits=self.k_folds, shuffle=True)

        for train, test in k_fold.split(X, y):
            model = self.get_model()
            model.fit(X[train], y[train], batch_size=self.batch_size, epochs=self.num_epochs, verbose=True)
            scores = model.evaluate(X[test], y[test], verbose=0)
            loss_history.append(scores)
            model_history.append(model)

        for i in range(0, len(loss_history)):
            print('****************************************************************************')
            print(f' fold {i + 1} - Loss: {loss_history[i]}')

        self.avg_loss = np.mean(loss_history)
        print('****************************************************************************')
        print(f' Average Loss: {self.avg_loss}')
        print('****************************************************************************')
        print('****************************************************************************')


        best_model_index = np.argmin(loss_history)
        best_model = model_history[best_model_index]
        self.model = best_model
        self.best_loss = loss_history[best_model_index]
        return best_model

    def evaluate_model(self, X, y):
        pred_y = self.model.predict(X)
        d = X.shape[0]
        fig = plt.figure(dpi=100)
        axs = fig.subplots(d, 3)

        for i in range(d):
            axs[i, 0].imshow(X[i].reshape(28, 28))
            axs[i, 1].imshow(pred_y[i].reshape(28, 28))
            axs[i, 2].imshow(y[i].reshape(28, 28))

        axs[0, 0].set_title("Noisy")
        axs[0, 1].set_title("DeNoised")
        axs[0, 2].set_title("Original")
        return fig


if __name__ == '__main__':
    dataset = NoisyDataset(noise_level=NoisyDataset.VERY_HIGH, train_n_samples=10000, test_n_samples=5)
    train_set = dataset.get_train_dataset()
    test_set = dataset.get_test_dataset()

    model = DigitImageDeNoise(batch_size=32, num_epochs=10)
    model.train(X=train_set[0], y=train_set[1])
    fig = model.evaluate_model(X=test_set[0], y=test_set[1])
    plt.show()
    fig.savefig('part_6.png')
