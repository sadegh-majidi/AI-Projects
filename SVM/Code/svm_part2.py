import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tensorflow import keras


def k_fold_svm_classification(X, y, kernel, k_folds=5, C=1, gamma='auto'):
    clf = SVC(kernel=kernel, C=C, gamma=gamma)
    history = cross_validate(
        clf, X, y, cv=k_folds, scoring='accuracy', return_estimator=True, return_train_score=True
    )
    ests = history['estimator']
    train_scores = history['train_score']
    test_scores = history['test_score']

    print('------------------------------------------------')
    print('train_scores: ')
    print(train_scores)
    print()
    print('test_scores: ')
    print(test_scores)
    print('------------------------------------------------')
    best_index = np.argmax(test_scores)

    return ests[best_index], train_scores[best_index]


if __name__ == '__main__':
    train_size = 50000
    test_size = 5000
    try:
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    except:
        with np.load('mnist.npz', allow_pickle=True) as f:
            X_train, y_train = f['x_train'], f['y_train']
            X_test, y_test = f['x_test'], f['y_test']
    y = np.concatenate((y_train, y_test)).reshape(70000)
    X = np.concatenate((X_train, X_test)).reshape(70000, 784)
    X = X.astype('float32')
    X /= 255
    X, y = shuffle(X, y)

    classifier, score = k_fold_svm_classification(X[:train_size], y[:train_size], k_folds=3, kernel='linear', C=1)
    pred_y = classifier.predict(X[train_size:train_size+test_size])
    print(f'test score {accuracy_score(y[train_size:train_size+test_size], pred_y)}')
    print(f'best train score {score}')
