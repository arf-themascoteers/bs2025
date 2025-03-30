import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


class DSManager:
    def __init__(self, name, remove_bg=True, test=False):
        self.name = name
        self.test=test
        dataset_path = f"data/{name}.csv"
        df = pd.read_csv(dataset_path)
        df.iloc[:, :-1] = minmax_scale(df.iloc[:, :-1])
        self.data = df.to_numpy()
        self.bs_train = self.data[self.data[:, -1] != 0]
        self.bs_train[:, -1] = self.bs_train[:, -1] - 1
        print(f"{self.name}: Total bs train samples", len(self.bs_train))

    def get_name(self):
        return self.name

    def get_train_data(self):
        return self.bs_train

    def get_train_x_y(self):
        return self.get_train_x(), self.get_train_y()

    def get_train_x(self):
        return self.bs_train[:,0:-1]

    def get_train_y(self):
        return self.bs_train[:, -1]

    def get_k_folds(self):
        folds = 20
        if self.test:
            folds = 1
        for i in range(folds):
            seed = 40 + i
            yield self.get_a_fold(seed)

    def get_a_fold(self, seed=50):
        return train_test_split(self.bs_train[:,0:-1], self.bs_train[:,-1], test_size=0.95, random_state=seed,
                         stratify=self.bs_train[:, -1])

    def __repr__(self):
        return self.get_name()

    @staticmethod
    def get_dataset_names():
        return [
            "indian_pines",
            "paviaU",
            "salinas"
        ]

