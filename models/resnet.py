import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10

from pytorch_lightning import LightningModule

from .metrics import Metrics


class ResNet18(LightningModule, Metrics):
    def __init__(self, num_classes=10):
        super().__init__()
        Metrics.__init__(self, num_classes=num_classes)
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.num_classes = num_classes
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        train_acc = self.train_accuracy_metric(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", train_acc, prog_bar=True, on_epoch=True)
        tensorboard_logs = {"train_loss": loss, "train_acc": train_acc}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        valid_acc = self.val_accuracy_metric(y_hat, y)
        precision_macro = self.precision_macro_metric(y_hat, y)
        recall_macro = self.recall_macro_metric(y_hat, y)
        precision_micro = self.precision_micro_metric(y_hat, y)
        recall_micro = self.recall_micro_metric(y_hat, y)
        f1 = self.f1_metric(y_hat, y)

        self.log("val_loss", loss, prog_bar=True, on_step=True)
        self.log("val_acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("val_precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("val_recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("val_precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("val_recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("val_f1", f1, prog_bar=True, on_epoch=True)

        tensorboard_logs = {
            "val_loss": loss,
            "val_acc": valid_acc,
            "val_precision_macro": precision_macro,
            "val_recall_macro": recall_macro,
            "val_precision_micro": precision_micro,
            "val_recall_micro": recall_micro,
            "val_f1": f1,
        }
        return {"val_loss": loss, "log": tensorboard_logs}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=3e-4)

    def train_dataloader(self):
        transform = ToTensor()
        train_dataset = CIFAR10(root="/tmp/CIFAR10", train=True, download=True, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        transform = ToTensor()
        val_dataset = CIFAR10(root="/tmp/CIFAR10", train=False, download=True, transform=transform)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        return val_dataloader
