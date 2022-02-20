import os

import matplotlib.pyplot as plt
import numpy as np
from keras.losses import sparse_categorical_crossentropy
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from tensorflow import keras

plt.style.use("ggplot")
os.environ['TF_KERAS'] = '1'


class DigitImageClassifier:
    def __init__(self, batch_size: int, num_epochs: int, k_folds: int = 5):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k_folds = k_folds
        self.loss_function = sparse_categorical_crossentropy
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.model = None
        self.best_loss = None
        self.avg_loss = None
        self.avg_acc = None

    def get_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(512, input_dim=28*28, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(10, activation='softmax'))

        model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def train(self, X, y):
        loss_history = []
        acc_history = []
        model_history = []
        k_fold = KFold(n_splits=self.k_folds, shuffle=True)

        for train, test in k_fold.split(X, y):
            model = self.get_model()
            model.fit(X[train], y[train], batch_size=self.batch_size, epochs=self.num_epochs, verbose=True)
            scores = model.evaluate(X[test], y[test], verbose=0)
            loss_history.append(scores[0])
            acc_history.append(scores[1]*100)
            model_history.append(model)

        for i in range(0, len(loss_history)):
            print('****************************************************************************')
            print(f' fold {i + 1} - Loss: {loss_history[i]} , Accuracy: {acc_history[i]}%')

        self.avg_loss = np.mean(loss_history)
        self.avg_acc = np.mean(acc_history)
        print('****************************************************************************')
        print(f' Average Loss: {self.avg_loss} , Average Accuracy: {self.avg_acc}%')
        print('****************************************************************************')
        print('****************************************************************************')


        best_model_index = np.argmin(loss_history)
        best_model = model_history[best_model_index]
        self.model = best_model
        self.best_loss = loss_history[best_model_index]
        return best_model

    def evaluate_model(self, X, y):
        self.model.evaluate(X, y)
        pred_y = self.model.predict(X)
        y_pred_bool = np.argmax(pred_y, axis=1)
        print(classification_report(y, y_pred_bool))
        self.plot_mis_classification_ex(X, y, y_pred_bool)

    def plot_mis_classification_ex(self, X, y_true, y_pred):
        for i in range(len(y_pred)):
            if y_pred[i] != y_true[i]:
                fig = plt.figure()
                image = X[i].reshape(28, 28)
                plt.imshow(image)
                plt.show()
                fig.savefig('part_5.png')
                print('\nMisClassified example:')
                print(f'Actual: {y_true[i]}')
                print(f'Predicted: {y_pred[i]}')
                break


if __name__ == '__main__':
    try:
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    except:
        with np.load('mnist.npz', allow_pickle=True) as f:
            X_train, y_train = f['x_train'], f['y_train']
            X_test, y_test = f['x_test'], f['y_test']
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    classifier = DigitImageClassifier(batch_size=128, num_epochs=6)
    classifier.train(X=X_train, y=y_train)
    classifier.evaluate_model(X=X_test, y=y_test)
