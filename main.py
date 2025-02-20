import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.load_data import get_data

if __name__ == "__main__":
    total_accuracies = []

    # n is the number of participants
    for n in range(2, 7):
        Xs, ys = [], []
        for i in range(1, 4):
            X_tmp, y_tmp = get_data(list(range(2, 2 + n)), i)
            Xs.append(X_tmp)
            ys.append(y_tmp)

        X = np.vstack(Xs)
        y = np.hstack(ys)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

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

        print(f'{"Name":^20} | {"Train accuracy":^15} | {"Test accuracy":^15}')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            for classificator, name in zip(classificators, names):
                classificator.fit(X_train, y_train)

                preds = classificator.predict(X_train)
                train_accuracy = accuracy_score(y_train, preds)

                preds = classificator.predict(X_test)
                test_accuracy = accuracy_score(y_test, preds)
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
