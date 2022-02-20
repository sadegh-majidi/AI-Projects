import time

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn.model_selection import train_test_split

diabetes_df = pd.read_csv('diabetes.csv')
y_label = 'Outcome'
test_size = 0.2

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(diabetes_df.drop(columns=[y_label]), diabetes_df[y_label],
                                                        test_size=test_size, shuffle=True)
    clf = DecisionTreeClassifier(criterion='entropy')

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    time.sleep(0.5)
    print(f"Accuracy {i} :", metrics.accuracy_score(y_test, y_pred))
