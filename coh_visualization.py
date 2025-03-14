import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.load_data import get_coh_data

# Xs, ys = [], []
# for i in range(1, 4):
#     X_tmp, y_tmp = get_coh_data(list(range(2, 13)), "fixing", "after")
#     Xs.append(X_tmp)
#     ys.append(y_tmp)

#     X_tmp, y_tmp = get_coh_data(list(range(2, 13)), "fixing", "before")
#     Xs.append(X_tmp)
#     ys.append(y_tmp)

# np.save("data/COH-data-X.npy", X)
# np.save("data/COH-data-y.npy", y)

plt.style.use("dark_background")

X = np.load("data/COH-data-X.npy")
y = np.load("data/COH-data-y.npy")

print(accuracy_score(y, SVC(kernel="linear").fit(X, y).predict(X)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# models = [
#     FastICA(n_components=2),
#     umap.UMAP(n_components=2, random_state=42, metric="correlation"),
#     PCA(n_components=2),
#     LinearDiscriminantAnalysis(n_components=2),
#     PLSRegression(n_components=2),
# ]
# names = ["ICA", "UMAP", "PCA", "LDA", "PLSRegression"]


# for model, name in zip(models, names):
#     model.fit(X_train, y_train)
#     X_ = model.transform(X_test)

#     for subject in set(y):
#         plt.scatter(X_[:, 0][y_test == subject], X_[:, 1][y_test == subject])

#     plt.title(name)
#     plt.savefig(f"coh_{name}.png", dpi=300)

#     plt.show()

X_ = tsne.fit_transform(X, y)
for subject in set(y_train):
    plt.scatter(X_[:, 0][y==subject],X_[:, 1][y==subject])
plt.show()
