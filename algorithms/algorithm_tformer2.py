from algorithm import Algorithm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math
import train_test_evaluator


class TFormer(nn.Module):
    def __init__(self, bands, number_of_classes):
        super().__init__()
        self.bands = bands
        self.number_of_classes = number_of_classes
        self.vector_length = 16
        self.embedding_layer = nn.Sequential(
            nn.Linear(1, self.vector_length),
            nn.LayerNorm(self.vector_length),
            nn.GELU()
        )
        self.mha = nn.MultiheadAttention(1, 1, batch_first=True)
        self.norm = nn.LayerNorm(1)
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.bands,1))
        self.base_band_attention = nn.Parameter(torch.ones(self.bands))
        self.band_attention = None
        self.fc_out = nn.Sequential(
            nn.Linear(self.bands, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16,self.number_of_classes)
        )
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, X, epoch):
        X_mod = X.view(X.size(0), X.size(1),1)
        X_mod = X_mod + self.pos_encoding
        attn_out, attn_weights = self.mha(X_mod, X_mod, X_mod, need_weights=True, average_attn_weights=False)
        attn_weights = attn_weights.mean(dim=1)
        attn_weights = attn_weights.mean(dim=1)
        attn_weights = attn_weights.mean(dim=0)
        self.band_attention = self.base_band_attention + attn_weights
        if epoch >=400 :
            dk = (180 - 50)/(500 - 400)*(epoch-400)
            k = int(180 - dk)
            print(k)
            threshold = torch.topk(self.band_attention, k).values[-1]
            self.band_attention[self.band_attention<threshold] = 0
        X = X*self.band_attention
        y = self.fc_out(X)
        return y, self.band_attention


class Algorithm_tformer2(Algorithm):
    def __init__(self, target_size: int, dataset, tag, reporter, verbose, test):
        super().__init__(target_size, dataset, tag, reporter, verbose, test)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.class_size = len(np.unique(self.dataset.get_train_y()))
        self.tformer = TFormer(self.dataset.get_train_x().shape[1], self.class_size).to(self.device)
        self.total_epoch = 500
        self.epoch = -1
        self.X_train = torch.tensor(self.dataset.get_train_x(), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.dataset.get_train_y(), dtype=torch.int32).to(self.device)

    def get_selected_indices(self):
        optimizer = torch.optim.Adam(self.tformer.parameters(), lr=0.001, betas=(0.9, 0.999))
        dataset = TensorDataset(self.X_train, self.y_train)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0
        lambda_value = 0.01
        for epoch in range(self.total_epoch):
            self.epoch = epoch
            weights = []
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                y_hat, channel_weights = self.tformer(X, epoch)
                weights.append(channel_weights.detach().cpu())
                y = y.type(torch.LongTensor).to(self.device)
                mse_loss = self.criterion(y_hat, y)
                l1_loss = self.l1_loss(channel_weights)
                lambda_value = 0.01#self.get_lambda(epoch + 1)
                loss = mse_loss + lambda_value * l1_loss
                loss.backward()
                optimizer.step()
            weights = torch.stack(weights)
            weights = weights.mean(dim=0)
            all_bands, selected_bands = self.get_band_sequence()
            self.set_all_indices(all_bands)
            self.set_selected_indices(selected_bands)
            self.set_weights(channel_weights)
            if self.epoch % 10 == 0:
                self.report_stats(channel_weights, epoch, mse_loss, l1_loss.item(), lambda_value, loss)

        print(self.get_name(), "selected bands and weights:")
        print("".join([str(i).ljust(10) for i in self.selected_indices]))
        return self.tformer, self.selected_indices

    def report_stats(self, channel_weights, epoch, mse_loss, l1_loss, lambda1, loss):
        min_cw = torch.min(channel_weights).item()
        max_cw = torch.max(channel_weights).item()
        avg_cw = torch.mean(channel_weights).item()

        l0_cw = torch.norm(channel_weights, p=0).item()

        all_bands, selected_bands = self.get_band_sequence()

        oa, aa, k = 0, 0, 0

        if self.verbose:
            oa, aa, k = train_test_evaluator.evaluate_split(*self.dataset.get_a_fold(), self)

        self.reporter.report_epoch(epoch, mse_loss, l1_loss, lambda1, loss,
                                   oa, aa, k,
                                   min_cw, max_cw, avg_cw,
                                   l0_cw,
                                   selected_bands, channel_weights)

    def get_band_sequence(self):
        band_indx = (torch.argsort(self.tformer.band_attention, descending=True)).tolist()
        return band_indx, band_indx[: self.target_size]

    def l1_loss(self, channel_weights):
        return torch.norm(channel_weights, p=1) / torch.numel(channel_weights)

    def get_lambda(self, epoch):
        return 0.0001 * math.exp(-epoch / self.total_epoch)



