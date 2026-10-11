import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import *

iris = load_iris()
X = iris.data
y = np.where(iris.target == 0, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
metrics = [[], [], [], []]
names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

for k in kernels:
    model = SVC(kernel=k)
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    scores = [accuracy_score(y_test, p),
              precision_score(y_test, p),
              recall_score(y_test, p), f1_score(y_test, p)]

    print(f"\nkernel:{k}")
    for n, s in zip(names, scores):
        print(f"{n}:{s:.4f}")

    for i in range(4):
        metrics[i].append(scores[i])

x = np.arange(4)
for i in range(4):
    plt.bar(x + (i-1.5)*0.2, metrics[i], 0.2, label=names[i])

plt.xticks(x, kernels)
plt.ylabel("Score")
plt.title("SVM Performance with Different Kernels")
plt.legend()
plt.show()
