import torch

from ds_manager import DSManager
from reporter import Reporter
import pandas as pd
from metrics import Metrics
from algorithm import Algorithm
import train_test_evaluator
import numpy as np


class TaskRunner:
    def __init__(self, task, tag="results", skip_all_bands=False, verbose=False, remove_bg=False, test=False):
        torch.manual_seed(3)
        self.task = task
        self.skip_all_bands = skip_all_bands
        self.verbose = verbose
        self.remove_bg = remove_bg
        self.test = test
        self.tag = tag
        self.reporter = Reporter(self.tag, self.skip_all_bands)
        self.cache = pd.DataFrame(columns=["dataset","algorithm","cache_tag","oa","aa","k","time","selected_features","selected_weights"])

    def evaluate(self):
        for dataset_name in self.task["datasets"]:
            dataset = DSManager(name=dataset_name, remove_bg=self.remove_bg, test=self.test)
            if not self.skip_all_bands:
                self.evaluate_for_all_features(dataset)
            for algorithm in self.task["algorithms"]:
                for target_size in self.task["target_sizes"]:
                    print(dataset_name, algorithm, target_size)
                    algorithm_object = Algorithm.create(algorithm, target_size, dataset, self.tag, self.reporter, self.verbose, self.test)
                    self.process_a_case(algorithm_object)

        self.reporter.save_results()
        return self.reporter.get_summary(), self.reporter.get_details()

    def process_a_case(self, algorithm:Algorithm):
        metric = self.reporter.get_saved_metrics(algorithm)
        if metric is None:
            oas, aas, ks, metric = self.get_results_for_a_case(algorithm)
            self.reporter.write_summary(algorithm, oas, aas, ks, metric)
        else:
            print(algorithm.get_name(), "for", algorithm.dataset.get_name(), "for",
                  algorithm.target_size,"was done. Skipping")

    def get_results_for_a_case(self, algorithm:Algorithm):
        metric = self.get_from_cache(algorithm)
        if metric is not None:
            print(f"Selected features got from cache for {algorithm.dataset.get_name()} "
                  f"for size {algorithm.target_size} "
                  f"for {algorithm.get_name()} "
                  f"for cache_tag {algorithm.get_cache_tag()}")
            algorithm.set_selected_indices(metric.selected_features)
            algorithm.set_weights(metric.selected_weights)
            return algorithm.compute_performance()
        print(f"NOT FOUND in cache for {algorithm.dataset.get_name()} "
              f"for size {algorithm.target_size} "
              f"for {algorithm.get_name()} "
              f"for cache_tag {algorithm.get_cache_tag()}. Computing.")
        oas, aas, ks, metric = algorithm.compute_performance()
        self.save_to_cache(algorithm, metric)
        return oas, aas, ks, metric

    def save_to_cache(self, algorithm:Algorithm, metric:Metrics):
        if not algorithm.is_cacheable():
            return
        self.cache.loc[len(self.cache)] = {
            "dataset":algorithm.dataset.get_name(),
            "algorithm": algorithm.get_name(),
            "cache_tag": algorithm.get_cache_tag(),
            "time":metric.time,"oa":metric.oa,"aa":metric.aa,"k":metric.k,
            "selected_features":algorithm.get_all_indices(),
            "selected_weights":algorithm.get_weights()
        }

    def get_from_cache(self, algorithm:Algorithm):
        if not algorithm.is_cacheable():
            return None
        if len(self.cache) == 0:
            return None
        rows = self.cache.loc[
            (self.cache["dataset"] == algorithm.dataset.get_name()) &
            (self.cache["algorithm"] == algorithm.get_name()) &
            (self.cache["cache_tag"] == algorithm.get_cache_tag())
        ]
        if len(rows) == 0:
            return None
        row = rows.iloc[0]
        selected_features = row["selected_features"][0:algorithm.target_size]
        selected_weights = row["selected_weights"][0:algorithm.target_size]
        return Metrics(row["time"], row["oa"],row["aa"], row["k"], selected_features, selected_weights)

    def evaluate_for_all_features(self, dataset):
        for fold, (train_x, test_x, train_y, test_y) in enumerate(dataset.get_k_folds()):
            oa, aa, k = train_test_evaluator.evaluate_split(train_x, test_x, train_y, test_y)
            self.reporter.write_details_all_features(fold, dataset.get_name(), oa, aa, k)


