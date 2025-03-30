from algorithm import Algorithm
import torch



class Algorithm_random(Algorithm):
    def __init__(self, target_size:int, dataset, tag, reporter, verbose, test):
        super().__init__(target_size, dataset, tag, reporter, verbose, test)
        self.indices = None

    def get_selected_indices(self):
        original_size = self.dataset.get_train_x().shape[1]
        self.indices = torch.randperm(original_size)[:self.target_size].sort().values.tolist()
        return self, self.indices

    def transform(self, X):
        return X[:,self.indices]

    def is_cacheable(self):
        return False
