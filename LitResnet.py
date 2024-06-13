from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torchmetrics.functional.classification import accuracy
import torchvision
from typing import Any


class LitResnet(LightningModule):
    def __init__(
        self, optimizer: dict[str, Any], batch_size: int = 256, label_smoothing=0.01
    ):
        super().__init__()

        self.save_hyperparameters()
        self.model = torchvision.models.resnet18(pretrained=False, num_classes=10)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        if self.hparams.optimizer.name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.optimizer.lr,
                betas=(self.hparams.optimizer.beta1, self.hparams.optimizer.beta2),
                weight_decay=self.hparams.optimizer.weight_decay,
                eps=self.hparams.optimizer.eps,
            )
        elif self.hparams.optimizer.name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.optimizer.lr,
                momentum=self.hparams.optimizer.momentum,
                weight_decay=self.hparams.optimizer.weight_decay,
                nesterov=self.hparams.optimizer.nesterov,
            )
        else:
            raise ValueError("Invalid optimizer name")
        return optimizer
