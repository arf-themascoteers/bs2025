from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import numpy as np



def average_accuracy(y_true, y_pred):
    ca = []
    for c in np.unique(y_true):
        y_c = y_true[np.nonzero(y_true == c)]
        y_c_p = y_pred[np.nonzero(y_true == c)]
        acurracy = accuracy_score(y_c, y_c_p)
        ca.append(acurracy)
    ca = np.array(ca)
    aa = ca.mean()
    return aa


def evaluate_train_test_pair(train_x, test_x, train_y, test_y):
    evaluator_algorithm = get_metric_evaluator()
    evaluator_algorithm.fit(train_x, train_y)
    y_pred = evaluator_algorithm.predict(test_x)
    return calculate_metrics(test_y, y_pred)


def evaluate_split(train_x, test_x, train_y, test_y, transform=None):
    if transform is not None:
        train_x = transform.transform(train_x)
        test_x = transform.transform(test_x)
    return evaluate_train_test_pair(train_x, test_x, train_y, test_y)


def calculate_metrics(y_test, y_pred):
    oa = accuracy_score(y_test, y_pred)
    aa = average_accuracy(y_test, y_pred)
    k = cohen_kappa_score(y_test, y_pred)
    return oa, aa, k


def get_metric_evaluator():
    gowith = "sv"

    if gowith == "rf":
        return RandomForestClassifier()
    elif gowith == "sv":
        return SVC(C=1e5, kernel='rbf', gamma=1.)
    else:
        return MLPClassifier(max_iter=2000)