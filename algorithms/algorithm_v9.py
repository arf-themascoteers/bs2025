from algorithm import Algorithm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math
import train_test_evaluator


class Sparse(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.last_k = 0

    def forward(self, X, epoch,l0_norm):
        self.last_k = self.get_k(epoch,l0_norm)
        X = torch.where(torch.abs(X) < self.last_k, 0, X)
        return X

    def get_k(self, epoch,l0_norm):
        l0_norm_threshold = 40
        start = 250
        maximum = 1
        end = 500
        minimum = 0

        if self.dataset == "indian_pines":
            l0_norm_threshold = 50

        if l0_norm <= l0_norm_threshold:
            return self.last_k


        if epoch < start:
            return minimum
        elif epoch > end:
            return maximum
        else:
            return (epoch - start) * (maximum / (end - start))


class ZhangNet(nn.Module):
    def __init__(self, bands, number_of_classes, last_layer_input, dataset):
        super().__init__()
        self.dataset = dataset
        self.bands = bands
        self.number_of_classes = number_of_classes
        self.last_layer_input = last_layer_input
        self.weighter = nn.Sequential(
            nn.Linear(self.bands, 512),
            nn.ReLU(),
            nn.Linear(512, self.bands)
        )
        self.classnet = nn.Sequential(
            nn.Linear(self.bands, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, self.number_of_classes),
        )
        self.sparse = Sparse(self.dataset)
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, X, epoch, l0_norm):
        channel_weights = self.weighter(X)
        channel_weights = torch.abs(channel_weights)
        channel_weights = torch.mean(channel_weights, dim=0)
        sparse_weights = self.sparse(channel_weights, epoch, l0_norm)
        reweight_out = X * sparse_weights
        output = self.classnet(reweight_out)
        return channel_weights, sparse_weights, output


class Algorithm_v9(Algorithm):
    def __init__(self, target_size:int, dataset, tag, reporter, verbose, test):
        super().__init__(target_size, dataset, tag, reporter, verbose, test)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.class_size = len(np.unique(self.dataset.get_train_y()))
        self.last_layer_input = 100
        if self.dataset.name == "paviaU":
            self.last_layer_input = 48
        self.zhangnet = ZhangNet(self.dataset.get_train_x().shape[1], self.class_size, self.last_layer_input,
                                 self.dataset.get_name()).to(self.device)
        self.total_epoch = 500
        self.epoch = -1
        self.X_train = torch.tensor(self.dataset.get_train_x(), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.dataset.get_train_y(), dtype=torch.int32).to(self.device)

    def get_selected_indices(self):
        optimizer = torch.optim.Adam(self.zhangnet.parameters(), lr=0.001, betas=(0.9,0.999))
        dataset = TensorDataset(self.X_train, self.y_train)
        dataloader = DataLoader(dataset, batch_size=12800000, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0
        l0_norm = self.X_train.shape[1]
        sparse_weights = None
        for epoch in range(self.total_epoch):
            self.epoch = epoch
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                if sparse_weights is None:
                    l0_norm = self.X_train.shape[1]
                else:
                    l0_norm = torch.norm(sparse_weights, p=0).item()
                channel_weights, sparse_weights, y_hat = self.zhangnet(X, epoch,l0_norm)
                deciding_weights = channel_weights
                mean_weight, all_bands, selected_bands = self.get_indices(deciding_weights)
                self.set_all_indices(all_bands)
                self.set_selected_indices(selected_bands)
                self.set_weights(mean_weight)

                y = y.type(torch.LongTensor).to(self.device)
                mse_loss = self.criterion(y_hat, y)
                l1_loss = self.l1_loss(channel_weights)
                lambda_value = self.get_lambda(l0_norm)
                loss = mse_loss + lambda_value*l1_loss
                if batch_idx == 0 and self.epoch%10 == 0:
                    self.report_stats(channel_weights, sparse_weights, epoch, mse_loss, l1_loss.item(), lambda_value,loss)
                loss.backward()
                optimizer.step()

        print(self.get_name(),"selected bands and weights:")
        print("".join([str(i).ljust(10) for i in self.selected_indices]))
        return self.zhangnet, self.selected_indices

    def report_stats(self, channel_weights, sparse_weights, epoch, mse_loss, l1_loss, lambda1, loss):
        mean_weight = channel_weights
        means_sparse = sparse_weights

        if len(mean_weight.shape) > 1:
            mean_weight = torch.mean(mean_weight, dim=0)
            means_sparse = torch.mean(means_sparse, dim=0)

        min_cw = torch.min(mean_weight).item()
        min_s = torch.min(means_sparse).item()
        max_cw = torch.max(mean_weight).item()
        max_s = torch.max(means_sparse).item()
        avg_cw = torch.mean(mean_weight).item()
        avg_s = torch.mean(means_sparse).item()

        l0_cw = torch.norm(mean_weight, p=0).item()
        l0_s = torch.norm(means_sparse, p=0).item()

        mean_weight, all_bands, selected_bands = self.get_indices(channel_weights)

        oa, aa, k = 0,0,0

        if self.verbose:
            oa, aa, k = train_test_evaluator.evaluate_split(*self.dataset.get_a_fold(), self)

        self.reporter.report_epoch(epoch, mse_loss, l1_loss, lambda1,loss,
                               oa, aa, k,
                               min_cw, max_cw, avg_cw,
                               min_s, max_s, avg_s,
                               l0_cw, l0_s,
                               selected_bands, means_sparse)

    def get_indices(self, deciding_weights):
        mean_weights = deciding_weights
        if len(mean_weights.shape) > 1:
            mean_weights = torch.mean(mean_weights, dim=0)

        corrected_weights = mean_weights
        if torch.any(corrected_weights < 0):
            corrected_weights = torch.abs(corrected_weights)

        band_indx = (torch.argsort(corrected_weights, descending=True)).tolist()
        return mean_weights, band_indx, band_indx[: self.target_size]

    def l1_loss(self, channel_weights):
        return torch.norm(channel_weights, p=1) / torch.numel(channel_weights)

    def get_lambda(self, l0_norm):
        l0_norm_threshold = 40
        if self.dataset == "indian_pines":
            l0_norm_threshold = 50
        if l0_norm <= l0_norm_threshold:
            return 0
        m = 0.001
        if self.dataset.get_name() == "salinas":
            m = 0.08
        elif self.dataset.get_name() == "indian_pines":
            m = 0.01
        return m



