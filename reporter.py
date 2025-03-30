import os
import pandas as pd
from metrics import Metrics
import torch
import shutil
import numpy as np


class Reporter:
    def __init__(self, tag="results", skip_all_bands=False):
        self.tag = tag
        self.skip_all_bands = skip_all_bands
        self.summary_filename = f"{tag}_summary.csv"
        self.details_filename = f"{tag}_details.csv"
        self.weight_filename = f"{tag}_weights_6.csv"
        self.save_dir = f"saved_results/{tag}"
        self.summary_file = os.path.join("results", self.summary_filename)
        self.details_file = os.path.join("results", self.details_filename)
        self.current_epoch_report_file = None
        self.current_weight_report_file = None
        self.current_weight_all_report_file = None
        os.makedirs("results", exist_ok=True)

        if not os.path.exists(self.summary_file):
            with open(self.summary_file, 'w') as file:
                file.write("dataset,target_size,algorithm,time,oa,aa,k,selected_features,selected_weights\n")

        if not os.path.exists(self.details_file):
            with open(self.details_file, 'w') as file:
                file.write("dataset,target_size,algorithm,oa,aa,k,fold\n")

        if self.skip_all_bands:
            return

        self.all_features_details_filename = f"{tag}_all_features_details.csv"
        self.all_features_summary_filename = f"{tag}_all_features_summary.csv"
        self.all_features_summary_file = os.path.join("results", self.all_features_summary_filename)
        self.all_features_details_file = os.path.join("results", self.all_features_details_filename)

        if not os.path.exists(self.all_features_summary_file):
            with open(self.all_features_summary_file, 'w') as file:
                file.write("dataset,oa,aa,k\n")

        if not os.path.exists(self.all_features_details_file):
            with open(self.all_features_details_file, 'w') as file:
                file.write("fold,dataset,oa,aa,k\n")

    def get_summary(self):
        return self.summary_file

    def get_details(self):
        return self.details_file

    def write_oak(self, algorithm, dataset, target_size, oa, aa, k):
        pass

    def write_summary(self, algorithm, oas, aas, ks, metric:Metrics):
        time = Reporter.sanitize_metric(metric.time)
        oa = Reporter.sanitize_metric(metric.oa)
        aa = Reporter.sanitize_metric(metric.aa)
        k = Reporter.sanitize_metric(metric.k)
        selected_features = np.array(metric.selected_features)
        selected_weights = np.array(metric.selected_weights)
        indices = np.argsort(selected_features)
        selected_features = selected_features[indices]
        selected_weights = selected_weights[indices]
        with open(self.summary_file, 'a') as file:
            file.write(f"{algorithm.dataset.get_name()},{algorithm.target_size},{algorithm.get_name()},"
                       f"{time},{oa},{aa},{k},"
                       f"{'|'.join([str(i) for i in selected_features])},"
                       f"{'|'.join([str(i) for i in selected_weights])}\n")

        with open(self.details_file, 'a') as file:
            for i in range(len(oas)):
                file.write(f"{algorithm.dataset.get_name()},{algorithm.target_size},{algorithm.get_name()},"
                       f"{round(oas[i],2)},{round(aas[i],2)},{round(ks[i],2)},{i}\n")

    def write_details_all_features(self, fold, name, oa, aa, k):
        oa = Reporter.sanitize_metric(oa)
        aa = Reporter.sanitize_metric(aa)
        k = Reporter.sanitize_metric(k)
        with open(self.all_features_details_file, 'a') as file:
            file.write(f"{fold},{name},{oa},{aa},{k}\n")
        self.update_summary_for_all_features(name)

    def update_summary_for_all_features(self, dataset):
        df = pd.read_csv(self.all_features_details_file)
        df = df[df["dataset"] == dataset]
        if len(df) == 0:
            return

        oa = round(max(df["oa"].mean(),0),2)
        aa = round(max(df["aa"].mean(),0),2)
        k = round(max(df["k"].mean(),0),2)

        df2 = pd.read_csv(self.all_features_summary_file)
        mask = (df2['dataset'] == dataset)
        if len(df2[mask]) == 0:
            df2.loc[len(df2)] = {"dataset":dataset, "oa":oa, "aa":aa, "k": k}
        else:
            df2.loc[mask, 'oa'] = oa
            df2.loc[mask, 'aa'] = aa
            df2.loc[mask, 'k'] = k
        df2.to_csv(self.all_features_summary_file, index=False)

    def get_saved_metrics(self, algorithm):
        df = pd.read_csv(self.summary_file)
        if len(df) == 0:
            return None
        rows = df.loc[(df["dataset"] == algorithm.dataset.get_name()) & (df["target_size"] == algorithm.target_size) &
                      (df["algorithm"] == algorithm.get_name())
                      ]
        if len(rows) == 0:
            return None
        row = rows.iloc[0]
        return Metrics(row["time"], row["oa"], row["aa"], row["k"], row["selected_features"], row["selected_weights"])

    def save_results(self):
        os.makedirs(self.save_dir, exist_ok=True)
        for filename in os.listdir("results"):
            if filename.startswith(f"{self.tag}_"):
                source_file = os.path.join("results", filename)
                if os.path.isfile(source_file):
                    shutil.copy(source_file, self.save_dir)

    @staticmethod
    def sanitize_metric(metric):
        if torch.is_tensor(metric):
            metric = metric.item()
        return round(max(metric, 0),3)

    @staticmethod
    def sanitize_weight(metric):
        if torch.is_tensor(metric):
            metric = metric.item()
        return round(metric,3)

    @staticmethod
    def sanitize_small(metric):
        if torch.is_tensor(metric):
            metric = metric.item()
        return round(metric,7)

    def create_epoch_report(self, tag, algorithm, dataset, target_size):
        self.current_epoch_report_file = os.path.join("results", f"{tag}_{algorithm}_{dataset}_{target_size}.csv")

    def create_weight_report(self, tag, algorithm, dataset, target_size):
        self.current_weight_report_file = os.path.join("results", f"{tag}_{algorithm}_{dataset}_{target_size}_weights.csv")

    def create_weight_all_report(self, tag, algorithm, dataset, target_size):
        self.current_weight_all_report_file = os.path.join("results", f"{tag}_{algorithm}_{dataset}_{target_size}_weights_all.csv")

    def report_epoch(self, epoch, mse_loss, l1_loss, lambda_value, loss,
                     oa,aa,k,
                     min_cw, max_cw, avg_cw,
                     min_s, max_s, avg_s,
                     l0_cw, l0_s,
                     selected_bands, mean_weight):
        if not os.path.exists(self.current_epoch_report_file):
            with open(self.current_epoch_report_file, 'w') as file:
                weight_labels = list(range(len(mean_weight)))
                weight_labels = [f"weight_{i}" for i in weight_labels]
                weight_labels = ",".join(weight_labels)
                file.write(f"epoch,"
                           f"l0_cw,l0_s,"
                           f"mse_loss,l1_loss,lambda_value,loss,"
                           f"oa,aa,k,"
                           f"min_cw,max_cw,avg_cw,"
                           f"min_s,max_s,avg_s,"
                           f"selected_bands,selected_weights,{weight_labels}\n")
        with open(self.current_epoch_report_file, 'a') as file:
            weights = [str(Reporter.sanitize_weight(i.item())) for i in mean_weight]
            weights = ",".join(weights)
            selected_bands_str = '|'.join([str(i) for i in selected_bands])

            selected_weights = [str(Reporter.sanitize_weight(i.item())) for i in mean_weight[selected_bands]]
            selected_weights_str = '|'.join(selected_weights)

            file.write(f"{epoch},"
                       f"{int(l0_cw)},{int(l0_s)},"
                       f"{Reporter.sanitize_metric(mse_loss)},"
                       f"{Reporter.sanitize_small(l1_loss)},{Reporter.sanitize_small(lambda_value)},"
                       f"{Reporter.sanitize_metric(loss)},"
                       f"{Reporter.sanitize_metric(oa)},{Reporter.sanitize_metric(aa)},{Reporter.sanitize_metric(k)},"
                       f"{Reporter.sanitize_weight(min_cw)},{Reporter.sanitize_weight(max_cw)},{Reporter.sanitize_weight(avg_cw)},"
                       f"{Reporter.sanitize_weight(min_s)},{Reporter.sanitize_weight(max_s)},{Reporter.sanitize_weight(avg_s)},"
                       f"{selected_bands_str},{selected_weights_str},{weights}\n")

    def report_weight(self, epoch, weights):
        if not os.path.exists(self.current_weight_report_file):
            with open(self.current_weight_report_file, 'w') as file:
                weight_labels = list(range(len(weights)))
                weight_labels = [f"weight_{i}" for i in weight_labels]
                weight_labels = ",".join(weight_labels)
                file.write(f"epoch,{weight_labels}\n")
        with open(self.current_weight_report_file, 'a') as file:
            weights = [str(Reporter.sanitize_weight(i.item())) for i in weights]
            weights = ",".join(weights)
            file.write(f"{epoch},{weights}\n")

    def report_weight_all(self, saved_weights):
        if not os.path.exists(self.current_weight_all_report_file):
            with open(self.current_weight_report_file, 'w') as file:
                file.write(f"batch,w1,w2,w3\n")
            with open(self.current_weight_all_report_file, 'a') as file:
                for i in range(0,500,10):
                    file.write(f"{i},{saved_weights[i,0].item()},{saved_weights[i,1].item()},{saved_weights[i,2].item()}\n")

