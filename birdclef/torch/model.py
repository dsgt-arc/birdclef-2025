import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics


class LinearClassifier(pl.LightningModule):
    def __init__(self, input_dim, num_classes, lr=1e-3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

        # Metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1_macro = torchmetrics.F1Score(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self.train_f1_micro = torchmetrics.F1Score(
            task="multiclass", average="micro", num_classes=num_classes
        )
        self.val_f1_macro = torchmetrics.F1Score(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self.val_f1_micro = torchmetrics.F1Score(
            task="multiclass", average="micro", num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        loss = self.loss_fn(logits, y)

        self.train_acc.update(preds, y)
        self.train_f1_macro.update(preds, y)
        self.train_f1_micro.update(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.log("train_f1_macro", self.train_f1_macro.compute(), prog_bar=True)
        self.log("train_f1_micro", self.train_f1_micro.compute(), prog_bar=True)

        self.train_acc.reset()
        self.train_f1_macro.reset()
        self.train_f1_micro.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        loss = self.loss_fn(logits, y)

        self.val_acc.update(preds, y)
        self.val_f1_macro.update(preds, y)
        self.val_f1_micro.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_f1_macro", self.val_f1_macro.compute(), prog_bar=True)
        self.log("val_f1_micro", self.val_f1_micro.compute(), prog_bar=True)

        self.val_acc.reset()
        self.val_f1_macro.reset()
        self.val_f1_micro.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
