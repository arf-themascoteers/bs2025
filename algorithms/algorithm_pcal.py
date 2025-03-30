from algorithm import Algorithm
from sklearn.decomposition import PCA
from auswahl import MCUVE

import numpy as np


class Algorithm_pcal(Algorithm):
    def __init__(self, target_size:int, dataset, tag, reporter, verbose, test):
        super().__init__(target_size, dataset, tag, reporter, verbose, test)

    def get_selected_indices(self):
        pca = PCA(n_components=self.target_size)
        pca.fit(self.dataset.get_train_x())
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        feature_importance = np.sum(np.abs(loadings), axis=1)
        feature_ranking = np.argsort(feature_importance)[::-1]
        indices = feature_ranking[:self.target_size]

        self.set_all_indices(feature_ranking)
        self.set_selected_indices(indices)
        self.set_weights(feature_importance[feature_ranking])
        return self, indices

