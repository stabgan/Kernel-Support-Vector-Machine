# Kernel SVM — Social Network Ads Classification
#
# Trains an RBF-kernel Support Vector Machine on the Social_Network_Ads
# dataset and visualises the decision boundary for both training and test sets.

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def load_dataset():
    """Load the Social Network Ads CSV relative to this script's location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "Social_Network_Ads.csv")
    dataset = pd.read_csv(csv_path)
    X = dataset.iloc[:, [2, 3]].values  # Age, EstimatedSalary
    y = dataset.iloc[:, 4].values       # Purchased
    return X, y


def plot_decision_boundary(X_set, y_set, classifier, title):
    """Plot the SVM decision boundary with scatter points."""
    colors = ("red", "green")
    cmap = ListedColormap(colors)

    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01),
    )
    Z = classifier.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)

    plt.contourf(X1, X2, Z, alpha=0.75, cmap=cmap)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for idx, label in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == label, 0],
            X_set[y_set == label, 1],
            c=colors[idx],
            label=label,
        )

    plt.title(title)
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.legend()
    plt.show()


def main():
    # Load data
    X, y = load_dataset()

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0,
    )

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fit Kernel SVM (RBF kernel)
    classifier = SVC(kernel="rbf", random_state=0)
    classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = classifier.predict(X_test)

    # Evaluation
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Purchased", "Purchased"]))

    # Visualise decision boundaries
    plot_decision_boundary(X_train, y_train, classifier, "Kernel SVM (Training set)")
    plot_decision_boundary(X_test, y_test, classifier, "Kernel SVM (Test set)")


if __name__ == "__main__":
    main()
