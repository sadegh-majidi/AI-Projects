import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


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


def draw(images, texts, columns=1, rows=1, image_size=28, scale=4):
    fig = plt.figure(figsize=(scale * columns, scale * rows))

    for i in range(columns * rows):
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.set_title(texts[i])
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(images[i].reshape(image_size, image_size) * 255)

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.set_frame_on(False)
    plt.show()


if __name__ == '__main__':
    dataset = np.load('persian_lpr.npz')
    X = dataset['images']
    y = dataset['targets']
    X = X.reshape((len(X), -1))
    X = X.astype('float32')
    X /= 255
    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    gammas = [0.01, 0.1, 1]
    Cs = [0.1, 1, 10]

    for i in range(3):
        for j in range(3):

            classifier, score = k_fold_svm_classification(X_train, y_train, k_folds=5, kernel='rbf', C=Cs[i], gamma=gammas[j])
            # classifier, score = k_fold_svm_classification(X_train, y_train, k_folds=5, kernel='linear', C=Cs[i], gamma=gammas[j])
            # classifier, score = k_fold_svm_classification(X_train, y_train, k_folds=5, kernel='poly', C=Cs[i], gamma=gammas[j])
            pred_y = classifier.predict(X_test)

            indices = np.random.choice(np.arange(len(X_test)), size=20)
            images = X_test[indices]
            predicted_digits = pred_y[indices]
            texts = [f'{predicted_digits[i]} - expected: {y_test[indices[i]]}' for i in range(len(indices))]
            draw(images, texts, 5, 4, image_size=16, scale=2)

            print(f'C = {Cs[i]}')
            print(f'gamma = {gammas[j]}')
            print(f'test score {accuracy_score(y_test, pred_y)}')
            print(f'best train score {score}')
            print('----------')
