import warnings

import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from itertools import chain

from src.load_data import extract_avg_features

if __name__ == "__main__":
    total_accuracies = []

    X_train, X_test = [], []
    y_train, y_test = [], []
    
    # They show when one class end for simpler access
    train_markers, test_markers = [], []

    # n is the index of participant
    for n in range(2, 4):
        # i is session number
        marker_train = 0
        for i in range(1, 4):
            data = read_csv(
                f"data/subject_{str(n).zfill(2)}_session_0{i}.csv"
            ).to_numpy()

            # let 3rd session be our test data
            X_, y_ = extract_avg_features(data, n)
            if i == 3:
                X_test.extend(X_)
                y_test.extend(y_)
                test_markers.append(len(X_))
            else:
                marker_train += len(X_)
                X_train.extend(X_)
                y_train.extend(y_)
                
        train_markers.append(marker_train)

    for n in range(2, 13):
        classificators = [
            KNeighborsClassifier(),
            LogisticRegression(max_iter=100),
            SVC(kernel="linear"),
            SVC(),
            DecisionTreeClassifier(),
            LinearDiscriminantAnalysis(),
            GaussianNB(),
        ]
        names = [
            "KNN",
            "Logistic Regression",
            "Linear SVM",
            "RBF SVM",
            "Decision Tree",
            "LDA",
            "Naïve Bayes",
        ]
        test_results = []

        X_train_, X_test_ = X_train[:int(np.cumsum(train_markers[:n]))], X_test[:int(np.cumsum(test_markers[:n]))]
        y_train_, y_test_ = y_train[:int(np.cumsum(train_markers[:n]))], y_test[:int(np.cumsum(test_markers[:n]))]
        
        y_train_ = np.array(y_train_)
        print(y_train_)

        # print(y_train[:int(np.cumsum(train_markers[:n-1]))])
        # print(y_test[:int(np.cumsum(test_markers[:n-1])))
        print(np.array(X_train_).shape)

        print(f'{"Name":^20} | {"Train accuracy":^15} | {"Test accuracy":^15}')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            for classificator, name in zip(classificators, names):
                print(list(set(y_train_)), list(set(y_test_)))
                classificator.fit(X_train_, y_train_)

                preds = classificator.predict(X_train_)
                train_accuracy = accuracy_score(y_train_, preds)

                preds = classificator.predict(X_test_)
                test_accuracy = accuracy_score(y_test_, preds)
                test_results.append(round(test_accuracy, 2))

                print(f"{name:>20} | {train_accuracy:^15.2f} | {test_accuracy:^15.2f}")

        total_accuracies.append(test_results)

    plt.style.use("dark_background")

    x = np.arange(len(names))
    width, multiplier = 0.15, -1

    fig, ax = plt.subplots(layout="constrained", figsize=(16, 9))
    for ind, accuracy in enumerate(total_accuracies):
        offset = width * multiplier
        rects = ax.bar(x + offset, accuracy, width, label=f"{ind+2}-classes")
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy for each classifier per different number of participants")
    ax.set_xticks(x + width, names)
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0.3, 1)
    plt.savefig("results.png", dpi=300)

    plt.show()
