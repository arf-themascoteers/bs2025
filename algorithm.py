from abc import ABC, abstractmethod
from metrics import Metrics
from datetime import datetime
import train_test_evaluator
import torch
import importlib
import numpy as np


class Algorithm(ABC):
    def __init__(self, target_size:int, dataset, tag, reporter, verbose, test=False):
        self.target_size = target_size
        self.dataset = dataset
        self.tag = tag
        self.reporter = reporter
        self.verbose = verbose
        self.test = test
        self.selected_indices = None
        self.weights = None
        self.model = None
        self.all_indices = None
        self.reporter.create_epoch_report(tag, self.get_name(), self.dataset.get_name(), self.target_size)
        self.reporter.create_weight_report(tag, self.get_name(), self.dataset.get_name(), self.target_size)
        self.reporter.create_weight_all_report(tag, self.get_name(), self.dataset.get_name(), self.target_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self):
        self.model, self.selected_indices = self.get_selected_indices()
        return self.selected_indices

    def transform(self, X):
        if len(self.selected_indices) != 0:
            return self.transform_with_selected_indices(X)
        return self.model.transform(X)

    def transform_with_selected_indices(self, X):
        return X[:,self.selected_indices]

    def compute_performance(self):
        start_time = datetime.now()
        if self.selected_indices is None:
            self.fit()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        oas = []
        aas = []
        ks = []
        for fold, (train_x, test_x, train_y, test_y) in enumerate(self.dataset.get_k_folds()):
            train_x = self.transform(train_x)
            test_x = self.transform(test_x)
            oa, aa, k = train_test_evaluator.evaluate_split(train_x, test_x, train_y, test_y)
            oas.append(oa)
            aas.append(aa)
            ks.append(k)
        oa = sum(oas) / len(oas)
        aa = sum(aas) / len(aas)
        k = sum(aas) / len(ks)
        return oas, aas, ks, Metrics(elapsed_time, oa, aa, k, self.selected_indices, self.get_weights())

    @abstractmethod
    def get_selected_indices(self):
        pass

    def get_name(self):
        class_name = self.__class__.__name__
        name_part = class_name[len("Algorithm_"):]
        return name_part

    def get_all_indices(self):
        return self.all_indices

    def set_all_indices(self, all_indices):
        self.all_indices = all_indices

    def set_selected_indices(self, selected_indices):
        self.selected_indices = selected_indices

    def is_cacheable(self):
        return True

    def get_cache_tag(self):
        return 0

    @staticmethod
    def create(name, target_size, dataset, tag, reporter, verbose, test):
        class_name = f"Algorithm_{name}"
        module = importlib.import_module(f"algorithms.algorithm_{name}")
        clazz = getattr(module, class_name)
        return clazz(target_size, dataset, tag, reporter, verbose, test)

    def set_weights(self, mean_weight):
        self.weights = mean_weight

    def get_weights(self):
        if torch.is_tensor(self.weights) or isinstance(self.weights, np.ndarray):
            return self.weights.tolist()
        return self.weights

