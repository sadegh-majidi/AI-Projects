import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_moons, make_blobs


def make_classification_dataset(size, low_bound, high_bound, criteria):
    dataset = pd.DataFrame({
            'x': np.random.uniform(low_bound, high_bound, size),
            'y': np.random.uniform(low_bound, high_bound, size),
            'class': -np.ones(size)
        })
    dataset['class'][criteria(dataset.x, dataset.y)] = 1
    return dataset


def plot_binary_svc(features, cls, svc):
    fig = plt.figure(figsize=(10, 10))
    if isinstance(features, pd.DataFrame):
        x, y = features['x'], features['y']
    else:
        x, y = features[:, 0], features[:, 1]
    plt.scatter(x, y, c=cls, s=50, cmap='spring')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svc.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=25, linewidth=1, label='support vectors',
               facecolors='none', edgecolors='k')
    plt.show()


def svm_classification(X, cls, kernel, C=1, gamma='auto', degree=3):
    model = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=1.5)
    model.fit(X, cls)
    plot_binary_svc(X, cls, model)


if __name__ == '__main__':
    TestNum = 52

    if TestNum // 10 == 0:
        data = make_classification_dataset(1000, -5, 5, lambda x, y: y - x <= 2)
        if TestNum % 10 == 0:
            svm_classification(data.drop('class', axis=1), data['class'], kernel='linear')
        elif TestNum % 10 == 1:
            svm_classification(data.drop('class', axis=1), data['class'], kernel='poly')
        elif TestNum % 10 == 2:
            svm_classification(data.drop('class', axis=1), data['class'], kernel='rbf')

    elif TestNum // 10 == 1:
        data = make_classification_dataset(1000, -1, 1, lambda x, y: x*y > 0)
        if TestNum % 10 == 0:
            svm_classification(data.drop('class', axis=1), data['class'], kernel='linear')
        elif TestNum % 10 == 1:
            svm_classification(data.drop('class', axis=1), data['class'], kernel='poly')
        elif TestNum % 10 == 2:
            svm_classification(data.drop('class', axis=1), data['class'], kernel='rbf')

    elif TestNum // 10 == 2:
        data = make_classification_dataset(1000, -5, 5, lambda x, y: (x**2) + (y**2)/2.25 <= 4)
        if TestNum % 10 == 0:
            svm_classification(data.drop('class', axis=1), data['class'], kernel='poly')
        elif TestNum % 10 == 1:
            svm_classification(data.drop('class', axis=1), data['class'], kernel='rbf')

    elif TestNum // 10 == 3:
        data = make_classification_dataset(2000, -5, 5, lambda x, y: (x**2) - (y**2) <= 1)
        if TestNum % 10 == 0:
            svm_classification(data.drop('class', axis=1), data['class'], kernel='rbf')
        elif TestNum % 10 == 1:
            svm_classification(data.drop('class', axis=1), data['class'], kernel='rbf', gamma=0.001)
        elif TestNum % 10 == 2:
            svm_classification(data.drop('class', axis=1), data['class'], kernel='rbf', gamma=100)

    elif TestNum // 10 == 4:
        data = make_classification_dataset(2000, -5, 5,
                                           lambda x, y: ((((x-1.5)**2) + ((y-1.5)**2) <= 3) & (((x-1.5)**2) + ((y-1.5)**2) >= 1))
                                           | ((((x+1.5)**2) + ((y+1.5)**2) <= 3) & (((x+1.5)**2) + ((y+1.5)**2) >= 1))
                                           )
        if TestNum % 10 == 0:
            svm_classification(data.drop('class', axis=1), data['class'], kernel='rbf')
        elif TestNum % 10 == 1:
            svm_classification(data.drop('class', axis=1), data['class'], kernel='rbf', C=0.001)
        elif TestNum % 10 == 2:
            svm_classification(data.drop('class', axis=1), data['class'], kernel='rbf', C=100)

    elif TestNum // 10 == 5:
        X, y = make_moons(n_samples=2000, noise=0.1, random_state=27)
        if TestNum % 10 == 0:
            svm_classification(X, y, kernel='linear')
        elif TestNum % 10 == 1:
            svm_classification(X, y, kernel='rbf')
        elif TestNum % 10 == 2:
            svm_classification(X, y, kernel='poly')
        elif TestNum % 10 == 3:
            svm_classification(X, y, kernel='rbf', gamma=100)

    elif TestNum // 10 == 6:
        X, y = make_blobs(n_samples=2000, centers=2, cluster_std=2, random_state=27)
        if TestNum % 10 == 0:
            svm_classification(X, y, kernel='linear')
        elif TestNum % 10 == 1:
            svm_classification(X, y, kernel='rbf')


