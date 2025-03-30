from algorithm import Algorithm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math
import train_test_evaluator


class TFormer(nn.Module):
    def __init__(self, bands, number_of_classes, last_layer_input):
        super().__init__()

        self.bands = bands
        self.number_of_classes = number_of_classes
        self.last_layer_input = last_layer_input
        self.weighter = nn.Sequential(
            nn.Linear(self.bands, 512),
            nn.ReLU(),
            nn.Linear(512, self.bands),
            nn.Sigmoid()
        )
        self.classnet = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(start_dim=1),
            nn.Linear(last_layer_input, self.number_of_classes)
        )
        self.sparse = Sparse()
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, X):
        channel_weights = self.weighter(X)
        sparse_weights = self.sparse(channel_weights)
        reweight_out = X * sparse_weights
        reweight_out = reweight_out.reshape(reweight_out.shape[0], 1, reweight_out.shape[1])
        output = self.classnet(reweight_out)
        return channel_weights, sparse_weights, output


class Algorithm_tformer1(Algorithm):
    def __init__(self, target_size: int, dataset, tag, reporter, verbose, test):
        super().__init__(target_size, dataset, tag, reporter, verbose, test)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.class_size = len(np.unique(self.dataset.get_train_y()))
        self.last_layer_input = 100
        self.bsformer = TFormer(self.dataset.get_train_x().shape[1], self.class_size, self.last_layer_input).to(self.device)
        self.total_epoch = 500
        self.epoch = -1
        self.X_train = torch.tensor(self.dataset.get_train_x(), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.dataset.get_train_y(), dtype=torch.int32).to(self.device)

    def get_selected_indices(self):
        optimizer = torch.optim.Adam(self.bsformer.parameters(), lr=0.001, betas=(0.9, 0.999))
        dataset = TensorDataset(self.X_train, self.y_train)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0

        for epoch in range(self.total_epoch):
            self.epoch = epoch
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                y_hat, channel_weights = self.bsformer(X)
                all_bands, selected_bands = self.get_band_sequence(channel_weights)
                self.set_all_indices(all_bands)
                self.set_selected_indices(selected_bands)
                self.set_weights(channel_weights)

                y = y.type(torch.LongTensor).to(self.device)
                mse_loss = self.criterion(y_hat, y)
                l1_loss = self.l1_loss(channel_weights)
                lambda_value = self.get_lambda(epoch + 1)
                loss = mse_loss + lambda_value * l1_loss
                if batch_idx == 0 and self.epoch % 10 == 0:
                    self.report_stats(channel_weights, epoch, mse_loss, l1_loss.item(), lambda_value,loss)
                loss.backward()
                optimizer.step()

        print(self.get_name(), "selected bands and weights:")
        print("".join([str(i).ljust(10) for i in self.selected_indices]))
        return self.bsformer, self.selected_indices

    def report_stats(self, channel_weights, epoch, mse_loss, l1_loss, lambda1, loss):
        min_cw = torch.min(channel_weights).item()
        max_cw = torch.max(channel_weights).item()
        avg_cw = torch.mean(channel_weights).item()

        l0_cw = torch.norm(channel_weights, p=0).item()

        all_bands, selected_bands = self.get_band_sequence(channel_weights)

        oa, aa, k = 0, 0, 0

        if self.verbose:
            oa, aa, k = train_test_evaluator.evaluate_split(*self.dataset.get_a_fold(), self)

        self.reporter.report_epoch(epoch, mse_loss, l1_loss, lambda1, loss,
                                   oa, aa, k,
                                   min_cw, max_cw, avg_cw,
                                   l0_cw,
                                   selected_bands)

    def get_band_sequence(self, channel_weights):
        band_indx = (torch.argsort(channel_weights, descending=True)).tolist()
        return band_indx, band_indx[: self.target_size]

    def l1_loss(self, channel_weights):
        return torch.norm(channel_weights, p=1) / torch.numel(channel_weights)

    def get_lambda(self, epoch):
        return 0.0001 * math.exp(-epoch / self.total_epoch)



