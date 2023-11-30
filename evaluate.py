import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import numpy as np
import torchsummary
import pathlib
import pickle
from utils import calc_class_weights, count_and_print_least_common_classes, load_ckpt, save_ckpt
from torchmetrics.classification import (MulticlassF1Score, MulticlassPrecision, 
                                            MulticlassRecall, MulticlassPrecisionRecallCurve,
                                            MulticlassROC, MulticlassConfusionMatrix, MulticlassAccuracy)
import time
from utils import save_obj, save_ckpt, load_ckpt


class evaluation():
    """
    Class for evaluating the model on the test set.
    This class assumes that the model is already trained and that we have the truth labels.
    In case we dont have the truth labels, we can just replace the metrics calculation and save the predictions.
    This would be provided in a separate class / file.
    """
    def __init__(self, model, test_loader, metrics, name) -> None:
        self.model = model
        self.test_loader = test_loader
        self.metrics = metrics
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for metric in self.metrics:
            self.metrics[metric].to(self.device)
        self.n_total_steps_test = len(self.test_loader)
        self.metrics_test_dict = { metric : 0 for metric in self.metrics}
    
    def add_to_metric(self, y_pred,y_true):

        y_true = torch.as_tensor(y_true, dtype=torch.float64)
        y_pred = torch.as_tensor(y_pred, dtype=torch.float64)

        for metric in self.metrics:
            self.metrics[metric].update(y_pred, y_true)

    def compute_metrics(self):
        #Compute metrics
        for metric in self.metrics:
            self.metrics_test_dict[metric] = self.metrics[metric].compute()

    def save_dicts(self):
        save_obj(self.metrics_test_dict, self.name +"/plots/metrics_dict_test" + "_" + str(self.name))

    def eval_batch(self):
        for i, (samples, labels) in enumerate(self.test_loader):
            samples = samples.to(self.device)
            labels = labels.to(self.device)
            y_pred = self.model(samples)
            self.add_to_metric(y_pred, labels)

            self.save_dicts()
            
        self.compute_metrics()

    def reset_metrics(self):
        for metric in self.metrics:
            self.metrics[metric].reset()

    def eval(self):
        with torch.no_grad():
            self.model.eval()
            self.eval_batch()
            self.save_dicts()
            self.reset_metrics()
            self.model.train()
        print("Evaluation done")
    


